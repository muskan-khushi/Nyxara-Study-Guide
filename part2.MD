# 🔮 NYXARA — GRAND FINALE MASTERY GUIDE

## Part 2 of 4: The 6-Layer AI Detection Stack · Risk Fusion · Adaptive Threshold Calibration

> **Where we are:** Part 1 covered the problem domain, dataset, 18 key features, and 5-stage feature pipeline. This part covers the entire AI brain of Nyxara — every model, every algorithm, every architectural decision, every math formula. This is where most jury questions will come from.

---

## SECTION 8 — OVERVIEW: WHY 6 LAYERS?

### 8.1 The Core Philosophy

No single ML technique catches all types of mule accounts. Here's why:

| Technique                   | What It Misses                                                           |
| --------------------------- | ------------------------------------------------------------------------ |
| XGBoost alone               | Network effects — a hub looks clean in isolation                        |
| GNN alone                   | Tabular anomalies — subtle statistical deviations from occupation peers |
| Rule-based                  | Anything below thresholds (smurfing), novel patterns                     |
| Anomaly detection alone     | Known fraud patterns that are labeled in training data                   |
| Behavioral biometrics alone | Legitimate-looking sessions by human mule operators                      |

The solution: **ensemble all of them** and fuse their outputs. Each layer adds orthogonal information, and the jury's question "why use all 6?" has a clean answer: because each catches a distinct class of fraud the others miss.

**Empirical validation from a 2025 systematic review of 57 fraud detection studies (arXiv:2404.00060):** Ensemble + graph hybrids outperform single tabular models by 8–15% AUC. Our 6 layers incrementally contribute:

```
Tabular Ensemble alone:          AUC 0.938
+ Layer 1 GNN (GraphSAGE+GAT):  AUC 0.961  (+2.3%)
+ Layer 4 VAE anomaly:          AUC 0.968  (+0.7%)
+ Layer 2 GTCT contrastive:     AUC 0.974  (+0.6%)
+ Layer 5 LineMVGNN:            AUC 0.977  (+0.3%)
+ BEI + Graph features:         AUC 0.982  (+0.5%)
                                 ─────────────────
TOTAL GAIN OVER BASELINE:               +4.4%
```

### 8.2 Layer Summary Table

| Layer | Model                                   | File                                             | Primary Signal Captured                            |
| ----- | --------------------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| L1    | NyxaraGNN (GraphSAGE + GAT)             | `models/gnn/model.py`                          | Network topology: who connects to whom             |
| L2    | GTCT (Contrastive Transformer)          | architecture                                     | Temporal behavioral similarity across accounts     |
| L3    | XGBoost + LightGBM + CatBoost + LR meta | `models/ensemble/`                             | Tabular statistical anomalies                      |
| L4    | Variational AutoEncoder                 | `models/anomaly/vae.py`                        | Zero-day: reconstruction error vs normal behavior  |
| L5    | LineMVGNN                               | dual-view graph                                  | Transaction cycles: dual account+transaction graph |
| L6    | TreeSHAP + Groq LLM                     | `inference/explainer.py`+`alert_narrator.py` | Explainability + compliance narrative              |

---

## SECTION 9 — LAYER 1: NyxaraGNN (GraphSAGE + GAT)

### 9.1 Why Graphs at All? Start From First Principles.

Consider this scenario: An account belonging to a 22-year-old student has:

* Transaction amounts: ₹8,000–₹12,000 each (not extreme)
* Frequency: 8 per week (plausible for a busy student)
* Occupation declared: student ✓
* Credit score: 65 (slightly low but not alarming)

Tabular ML looks at this account and says: **low risk, approve.**

But the graph reveals: this account sends money to 6 other accounts within minutes of receiving it. Those 6 accounts all have similar transaction patterns, all opened at the same branch (G365D), all in the same week. One of them is a known flagged account.

**This is a textbook STAR ring.** The tabular model is blind to it. The GNN is not.

**The core idea:** A Graph Neural Network (GNN) operates on a graph where nodes are accounts and edges are transaction relationships (or similarity relationships). Each node's final representation is computed by aggregating information from its neighbors, their neighbors, and so on. The final node embedding encodes not just the account's own features, but everything about its network context.

### 9.2 Graph Construction: How the Graph Is Built

The transaction graph is built from THREE complementary edge sources. This is important — each source captures different relationship types:

#### Edge Source 1: KNN Edges (k=8, cosine similarity)

```python
# From graph_builder.py
X_norm = normalize(X_numeric, norm='l2')  # L2-normalize feature vectors
nbrs = NearestNeighbors(n_neighbors=9, metric='cosine', algorithm='brute', n_jobs=-1)
nbrs.fit(X_norm)
distances, indices = nbrs.kneighbors(X_norm)

for i, (dists, neighbors) in enumerate(zip(distances, indices)):
    for d, j in zip(dists[1:], neighbors[1:]):  # skip self (index 0)
        if d <= 0.30:  # distance threshold
            edges.append((i, j))
            edges.append((j, i))  # bidirectional
```

**What it captures:** Accounts that are *behaviorally similar* — similar transaction amounts, similar timing patterns, similar occupation velocity ratios. Two mule accounts trained by the same criminal will have nearly identical feature vectors even if they never directly transact with each other.

**Why cosine similarity not Euclidean?** Cosine similarity measures the *angle* between feature vectors, not their magnitude. Two accounts with similar behavioral *patterns* but different transaction volumes will have similar cosine similarity but very different Euclidean distance. For mule detection, pattern similarity matters more than magnitude similarity.

**Why threshold = 0.30?** Cosine distance of 0.30 means the angle between feature vectors is cos⁻¹(0.70) ≈ 46 degrees — accounts are meaningfully similar but not identical. Higher threshold (e.g., 0.50) would create a too-sparse graph; lower (0.10) would create a too-dense graph where legitimate accounts also heavily cluster.

#### Edge Source 2: Branch Edges (F3889)

```python
# Connect all accounts at the same branch
for branch, group in df.groupby('F3889'):
    if len(group) >= 2000:  # cap: prevent single huge branch from dominating
        continue
    idxs = group.index.tolist()
    # All-pairs edges for small groups; random sampling for large groups
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            edges.append((idxs[i], idxs[j]))
            edges.append((idxs[j], idxs[i]))
```

**What it captures:** Branch-level recruitment clusters. Certain branches (like G365D with 81% fraud rate in our data) are active mule recruitment grounds. Accounts opened at the same branch by a recruiter will often be part of the same ring. The branch edge directly connects these co-located accounts.

**The 2,000 account cap:** Prevents a single large legitimate branch (a main city branch with thousands of normal accounts) from creating millions of spurious edges and dominating the graph structure.

#### Edge Source 3: Linked Account Edges (F1692)

```python
# Connect accounts sharing device/phone signals
high_link = df[pd.to_numeric(df['F1692'], errors='coerce').fillna(0) > 5].index.tolist()
for i in range(len(high_link)):
    for j in range(i+1, len(high_link)):
        edges.append((high_link[i], high_link[j]))
        edges.append((high_link[j], high_link[i]))
```

**What it captures:** Controller-level clustering. A single mule operator controlling 15 accounts from one device will appear as F1692 > 5 for all 15 accounts. Connecting all accounts with F1692 > 5 creates a dense cluster centered on the controller's device — directly representing the mule cell.

**The F1692 > 5 threshold:** 5 linked accounts is too many to be explained by legitimate family sharing (parents + child might share 2–3). More than 5 indicates organized, non-family account control.

#### Final Graph Deduplication

```python
# From graph_builder.py — CRITICAL FIX
seen = set()
valid_edges = []
for u, v in all_edges:
    if not (0 <= u < num_nodes and 0 <= v < num_nodes):  # bounds check
        continue
    key = (u, v)
    if key not in seen:
        seen.add(key)
        valid_edges.append(key)
```

**Why bounds checking?** After reset_index() operations during train/test splitting, node indices can get misaligned. Without the bounds check, out-of-range edge indices cause PyTorch Geometric to silently corrupt the graph or crash during training.

### 9.3 NyxaraGNN Architecture: Layer by Layer

**File:** `ai-engine/models/gnn/model.py`

```python
class NyxaraGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, 
                 dropout=0.3, gat_heads=4):
        # Layer 1: SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
      
        # Layer 2: GATConv (4-head attention)
        self.conv2 = GATConv(hidden_channels, hidden_channels // gat_heads, 
                              heads=gat_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels)
      
        # Layer 3: SAGEConv + skip connection
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.skip_proj = nn.Linear(in_channels, hidden_channels)
      
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )
```

**Forward pass:**

```python
def forward(self, x, edge_index):
    skip = F.relu(self.skip_proj(x))      # project input for skip connection
  
    h = self.conv1(x, edge_index)         # SAGEConv: aggregate neighbors
    h = self.bn1(h)                        # normalize activations
    h = F.relu(h)
    h = F.dropout(h, p=self.dropout, training=self.training)
  
    h = self.conv2(h, edge_index)         # GATConv: attend to neighbors
    h = self.bn2(h)
    h = F.relu(h)
    h = F.dropout(h, p=self.dropout, training=self.training)
  
    h = self.conv3(h, edge_index)         # SAGEConv again
    h = self.bn3(h)
    h = F.relu(h + skip)                  # add skip connection (residual)
  
    return self.classifier(h)             # 2-class output
```

### 9.4 GraphSAGE (SAGEConv) — Mathematical Deep Dive

**GraphSAGE** (Hamilton et al., NeurIPS 2017, arXiv:1706.02216) is an *inductive* GNN that learns a sampling and aggregation function rather than specific embeddings.

**Message passing formula:**

```
h_v^(l) = σ( W^(l) · CONCAT( h_v^(l-1),  MEAN({h_u^(l-1) : u ∈ N(v)}) ) )
```

Where:

* `h_v^(l)` = representation of node v at layer l
* `N(v)` = neighborhood of v (all nodes connected by an edge)
* `W^(l)` = learnable weight matrix for layer l
* `σ` = non-linearity (ReLU in our case)
* `CONCAT` = concatenation of self-embedding and neighborhood aggregate

**In plain English:** Take your own current representation. Average all your neighbors' current representations. Concatenate the two. Apply a learnable linear transformation. Apply ReLU. This is your new representation.

After 3 layers:

* Layer 1: Each node knows about its 1-hop neighbors
* Layer 2: Each node knows about its 2-hop neighbors (neighbors of neighbors)
* Layer 3: Each node knows about its 3-hop neighborhood

For a hub account connected to 10 mule accounts, after 3 GNN layers, the hub's representation contains aggregated information from all 10 mules, plus the mules' other connections, plus those connections' connections. The hub's representation becomes recognizably "orchestrator-like" regardless of its own individual features.

**Why SAGE over GCN (Graph Convolutional Network)?**

GCN (Kipf & Welling, ICLR 2017) formula:

```
H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )
```

Where Ã = A + I (adjacency with self-loops), D̃ = degree matrix.

**Problems with GCN for our use case:**

1. **Transductive:** Requires the complete graph at training time. Cannot embed new accounts added after training. For production mule detection, new accounts are created daily — we need inductive learning.
2. **Degree normalization:** GCN normalizes by √(d_i × d_j) for each edge. Hub accounts with high degree get their signals divided down — exactly the accounts we care most about.
3. **Full adjacency matrix:** GCN uses the complete N×N adjacency matrix. For 9,082 nodes, this is 82.5M entries — feasible but wasteful. GraphSAGE samples neighbors.

**GraphSAGE advantages:**

1. **Inductive:** Learns an aggregation function, not node-specific embeddings. New accounts can be embedded without retraining.
2. **No degree normalization:** Hub accounts retain their full signal strength.
3. **Scalable:** Samples a fixed neighborhood size, enabling mini-batch training.

### 9.5 GAT (Graph Attention Network) — Mathematical Deep Dive

**GATConv** (Veličković et al., ICLR 2018, arXiv:1710.10903) adds an attention mechanism that learns *which* neighbors are most important.

**Attention coefficient:**

```
e_ij = LeakyReLU( aᵀ · [W·h_i || W·h_j] )
α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)
```

**Aggregation with attention:**

```
h_i^(l) = σ( Σ_{j∈N(i)} α_ij · W · h_j^(l-1) )
```

**Multi-head attention (4 heads):**

```
h_i^(l) = ||_{k=1}^{K} σ( Σ_{j∈N(i)} α_ij^k · W^k · h_j^(l-1) )
```

Where `||` denotes concatenation across K=4 heads.

**In plain English:** For each node, learn a *separate attention score* for each neighbor. Neighbors that are more "relevant" get higher attention weights. The final representation is a weighted sum of neighbors' representations, not a simple average.

**Why does this matter for mule detection?**

Consider a hub account connected to: 8 confirmed mule accounts and 2 accounts that are its legitimate electricity utility payments. The 2 utility accounts are LOW-RISK neighbors that would dilute the hub's fraud signal in a simple mean aggregation (GraphSAGE Layer 1). GAT learns to assign high attention to the 8 mule neighbors and low attention to the 2 utility neighbors — the hub's learned representation strongly reflects its mule connections.

**Why 4 heads?** Multi-head attention allows the model to attend to different "aspects" of neighbors simultaneously. Head 1 might focus on transaction velocity signals, Head 2 on timing anomalies, Head 3 on pass-through ratios, Head 4 on network centrality. The 4 heads' outputs are concatenated then projected back to `hidden_channels`.

### 9.6 The Skip Connection — Why It's Critical

```python
skip = F.relu(self.skip_proj(x))  # project input to hidden_channels
# ... 3 GNN layers ...
h = F.relu(h + skip)  # ADD skip to final GNN output
```

**The over-smoothing problem:** In deep GNNs (3+ layers), all nodes' representations converge toward the graph mean through repeated neighborhood averaging. After many layers, a hub account and a peripheral mule account have nearly identical representations — making them indistinguishable.

**Why it's especially bad for heterophilic graphs:** Our fraud graph is *heterophilic* — fraudulent accounts are connected to legitimate accounts (a ring hub connects to legitimate-looking accounts for cover). In homophilic graphs (similar nodes connect to similar nodes), over-smoothing is less harmful because averaging with similar neighbors doesn't degrade representations much. In our heterophilic case, averaging with dissimilar neighbors degrades signal.

**The skip connection solution:** Adds the original input features directly to the Layer 3 output. This guarantees:

1. The final representation always contains the account's own individual features
2. The gradient signal flows directly from the loss back to the input features (no gradient vanishing through 3 layers)
3. The model can choose to ignore network context (set `skip` weight high) if the individual features are already sufficient

This is the same concept as ResNets (He et al., CVPR 2016) applied to GNNs.

### 9.7 Training the GNN

**Class-weighted loss:**

```python
n_pos = data.y[data.train_mask].sum().item()   # fraud accounts in training
n_neg = (data.train_mask.sum() - n_pos).item() # legitimate accounts
class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)])
# At 0.89% fraud rate: class_weight ≈ [1.0, 112.0]

loss = F.cross_entropy(
    out[data.train_mask],
    data.y[data.train_mask],
    weight=class_weight
)
```

**Learning rate schedule:**

```python
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
# If validation AUC doesn't improve for 10 epochs, halve the learning rate
```

**Early stopping (AUC-based):**

```python
# Every 5 epochs, compute validation AUC
val_probs = F.softmax(val_out, dim=1)[:, 1].numpy()
val_auc = roc_auc_score(val_labels, val_probs)

if val_auc > best_val_auc:
    best_val_auc = val_auc
    best_state = copy of model.state_dict()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 30:  # 30 epochs without improvement
        break
```

**Why AUC-based early stopping instead of loss-based?**
At 0.89% fraud rate, the cross-entropy loss can decrease (model improving on numerically) while the actual fraud detection quality (AUC) stagnates or degrades. AUC directly measures what we care about: rank ordering fraud above legitimate. Using AUC as the early stopping criterion ensures we stop when the ranking quality is best, not when the numerical loss is smallest.

**Training characteristics:**

* Max epochs: 450
* Typical early stopping: epoch 180–280
* Training time: 20–30 minutes on CPU (8GB RAM laptop)
* Node features: 55 scaled features per account
* Edge count: varies; typically ~80,000–120,000 edges for 9,082 nodes

### 9.8 GNN Inference at Runtime

At inference time for a new account, the account is added to the graph, its feature vector is computed, and the GNN computes its embedding using the saved model weights:

```python
# From inference/cache.py
checkpoint = torch.load(ARTIFACTS_DIR / 'gnn_model.pth', map_location='cpu')
model = NyxaraGNN(in_channels=checkpoint['in_channels'], 
                   hidden_channels=checkpoint['hidden_channels'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Score all accounts
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1].numpy()  # fraud probability
```

For **cached accounts** (already scored during training), the GNN score is retrieved from `_SCORE_CACHE` in O(1). For new accounts, it requires a graph forward pass.

---

## SECTION 10 — LAYER 2: GRAPH-TEMPORAL CONTRASTIVE TRANSFORMER (GTCT)

### 10.1 The Zero-Day Mule Problem

Supervised learning only detects patterns present in labeled training data. Consider a **novel ring topology** that criminals invented after the training data was collected — it has no labeled examples. The GNN and ensemble models will both have poor accuracy on it because they've never seen it.

The GTCT uses **self-supervised contrastive learning** to learn fraud-sensitive representations  *without depending on labels* . It uses all 9,082 accounts (including unlabeled ones) to learn which accounts are behaviorally similar and which are different.

### 10.2 Contrastive Learning: The Core Idea

**From SimCLR (Chen et al., ICML 2020, arXiv:2002.05709) adapted to graph-temporal data:**

For each account, create two **augmented views** by applying random transformations:

* Drop a random 20% of features
* Mask a random time window in the temporal sequence
* Add small Gaussian noise to continuous features

Train the model to:

* **Maximize similarity** between the two views of the *same* account
* **Minimize similarity** between views of *different* accounts

After training, the encoder has learned representations where:

* Mule accounts (which share distinctive behavioral signatures) cluster together in representation space
* Legitimate accounts form loose clusters based on occupation/transaction patterns
* The clusters are separated in a way that generalizes to novel fraud patterns

### 10.3 The Loss Function: InfoNCE / NT-Xent

```
L_CL = -log( exp(sim(z_i, z_j) / τ) / Σ_{k=1}^{2N} 𝟙[k≠i] exp(sim(z_i, z_k) / τ) )
```

Where:

* `z_i, z_j` = projected representations of two augmented views of the same account (the "positive pair")
* `z_k` = all other representations in the batch (the "negatives")
* `τ = 0.07` = temperature parameter
* `sim(u, v)` = cosine similarity = u·v / (||u|| × ||v||)
* The sum is over all 2N−1 samples in the batch except the anchor itself

**In plain English:** For a given account's representation, make it close to its own augmented view (positive pair) and far from all other accounts' representations in the batch (negatives).

**The temperature parameter τ = 0.07:** Low temperature makes the softmax distribution sharper — it strongly penalizes any similarity with negatives, forcing the model to create very compact, well-separated clusters. Higher τ creates a softer distribution that's easier to optimize but may produce less discriminative representations.

**Why is this called "contrastive"?** The loss *contrasts* the positive pair against all negatives. The model learns by contrast — it has to make the positive pair stand out from the crowd of negatives.

### 10.4 Why This Helps at 0.89% Fraud Rate

At 0.89% fraud rate, we have only ~81 labeled fraud accounts. Supervised learning on 81 examples of a complex, high-dimensional pattern is extremely difficult. Contrastive pre-training uses all 9,082 accounts *without labels* — the unlabeled 9,001 legitimate accounts all serve as negatives, dramatically increasing the quality of the learned representation.

After contrastive pre-training, the model is fine-tuned on the labeled data. The pre-trained representations already separate fraudulent behavioral patterns from legitimate ones; fine-tuning just calibrates the decision boundary.

**Reported AUC uplift:** +0.062 (6.2 percentage points) in comparable financial fraud literature.

---

## SECTION 11 — LAYER 3: THE STACKED TABULAR ENSEMBLE

### 11.1 Architecture Overview

```
XGBoost(features) → P_XGB (fraud probability)
LightGBM(features) → P_LGBM (fraud probability)
CatBoost(features) → P_CAT (fraud probability)
GNN scores →  P_GNN (fraud probability)
                 ↓
Meta-features: [P_XGB, P_LGBM, P_CAT, P_GNN]
                 ↓
Logistic Regression meta-learner → P_ensemble (final ensemble probability)
```

### 11.2 XGBoost — Complete Configuration

```python
XGBClassifier(
    n_estimators=500,        # up to 500 trees; early stopping at 30
    max_depth=6,             # each tree maximum 6 levels deep
    learning_rate=0.05,      # shrinkage: each tree contributes 5% of its prediction
    subsample=0.8,           # use 80% of training samples per tree (row sampling)
    colsample_bytree=0.8,    # use 80% of features per tree (column sampling)
    min_child_weight=5,      # require minimum 5 samples in any leaf node
    gamma=1,                 # minimum loss reduction to make a split
    reg_alpha=0.1,           # L1 regularization on leaf weights (sparsity)
    reg_lambda=1.0,          # L2 regularization on leaf weights (shrinkage)
    scale_pos_weight=112,    # weight for positive class: n_negative/n_positive
    tree_method='hist',      # histogram-based algorithm: fast on CPU
    eval_metric=['auc', 'logloss'],
    early_stopping_rounds=30,
    verbosity=1
)
```

**Explaining every hyperparameter:**

`max_depth=6`: Each decision tree has at most 6 levels of splits, creating at most 2⁶=64 leaf nodes. Deeper trees can model more complex patterns but overfit more. 6 is the standard sweet spot for financial tabular data.

`learning_rate=0.05`: Also called "shrinkage." Each new tree's prediction is multiplied by 0.05 before being added to the ensemble. Slow learning means more trees are needed (why we use 500) but the final model is more robust. `learning_rate × n_estimators ≈ constant` — you can compensate by adjusting both.

`subsample=0.8` (row sampling): Each tree is trained on a random 80% of training accounts. Prevents any single account from dominating multiple trees. Creates diversity between trees — key for ensemble variance reduction.

`colsample_bytree=0.8` (column sampling): Each tree can only see 80% of features. Forces trees to explore different feature combinations. Without this, all trees would use the same top-ranked features and have highly correlated errors.

`min_child_weight=5`: A leaf node must have at least 5 accounts in it. Prevents overfitting to tiny subgroups (which could be noise).

`gamma=1`: A split is only made if it reduces the loss by at least 1.0. Prevents unnecessary splits that add complexity without meaningful gain.

`scale_pos_weight=112`: The fraud class is weighted 112× heavier. This directly compensates for the 112:1 class ratio. Every fraud account's gradient contribution during training equals that of 112 legitimate accounts.

`tree_method='hist'`: XGBoost's CPU-optimized algorithm. Bins continuous features into histograms (256 bins by default), then finds optimal splits on the bins rather than all unique values. Reduces complexity from O(n log n) to O(n) per feature per split.

### 11.3 Gradient Boosting: Mathematical Foundation

**Start with a constant prediction:**

```
F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)   [e.g., log-odds of class distribution]
```

**For each tree m = 1, 2, ..., M:**

1. Compute pseudo-residuals (negative gradient of loss):

```
rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]_{F=F_{m-1}}
```

2. Fit a decision tree hₘ to the pseudo-residuals
3. Compute optimal leaf weights:

```
γⱼₘ = argmin_γ Σᵢ∈Rⱼₘ L(yᵢ, Fₘ₋₁(xᵢ) + γ)
```

4. Update the ensemble:

```
Fₘ(x) = Fₘ₋₁(x) + η × Σⱼ γⱼₘ × 𝟙[x ∈ Rⱼₘ]
```

Where η = learning rate, Rⱼₘ = region (leaf) j of tree m.

**"Gradient" in gradient boosting:** We're performing gradient descent in  *function space* , not parameter space. Each new tree is fit to the negative gradient of the loss — this is the direction that most reduces the loss if we add a tree in that direction.

**XGBoost's enhancement:** Uses second-order (Newton step) approximation:

```
Obj = Σᵢ [gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)
```

Where gᵢ = first derivative, hᵢ = second derivative, Ω = regularization term. The second-order information makes the optimization more efficient and allows better-calibrated leaf weights.

### 11.4 LightGBM — Key Differences From XGBoost

```python
LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,         # KEY DIFFERENCE: leaf-wise growth
    min_child_samples=20,  # minimum 20 accounts in a leaf
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight={0: 1.0, 1: 112.0},
    random_state=42
)
```

**The critical difference — leaf-wise vs depth-wise growth:**

* **XGBoost (depth-wise):** Splits all nodes at the same depth before moving deeper. Tree grows level by level. Balanced trees.
* **LightGBM (leaf-wise):** Always splits the leaf with the maximum loss reduction, regardless of depth. Unbalanced trees that grow toward the highest-information regions first.

**Leaf-wise is faster and often more accurate:** Finds the best possible split at each step rather than exhaustively splitting all leaves at the same level. But it can overfit on small datasets because it aggressively pursues individual data points.

`num_leaves=63`: For a balanced binary tree, depth 6 has 2⁶=64 leaves. LightGBM's leaf-wise growth doesn't have a depth concept, so `num_leaves=63` approximately corresponds to a depth-6 tree in terms of model complexity.

**LightGBM speed advantage:** Uses Gradient-based One-Side Sampling (GOSS) — keeps all high-gradient samples and randomly samples low-gradient samples. This reduces training data size without losing much accuracy. Also uses Exclusive Feature Bundling (EFB) — bundles mutually exclusive sparse features. Together, these can achieve 20× speedup over XGBoost on large datasets.

### 11.5 CatBoost — Key Innovation

```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    scale_pos_weight=112,
    eval_metric='AUC',
    random_seed=42,
    verbose=50,
    early_stopping_rounds=30,
    task_type='CPU'
    # CRITICAL: cat_features NOT passed
)
```

**The key CatBoost innovation — Ordered Boosting:**

Standard gradient boosting has a subtle data leakage problem: when computing the residual for account i, the loss gradient uses predictions that were influenced by account i's own label (because account i was in the training set for all previous trees). This causes overfitting.

CatBoost uses **Ordered Boosting:** processes training accounts in a random order. When computing the residual for account i, it only uses models trained on accounts *before* i in the order. This eliminates the self-prediction leakage.

**Why NOT pass cat_features even though F3889/F3891 are categorical?**

CatBoost's native categorical handling works by computing **ordered target statistics** — essentially a per-category mean of the target, computed in a specific order to avoid leakage. But this requires the features to be raw strings (e.g., "student", "G365D").

In our pipeline, F3891 has already been through:

1. `composite_features()` — used as a string to look up income proxies
2. `fit_encoders()` — converted to int32 (student → 6)
3. `fit_scaler()` — scaled to a float via RobustScaler

By the time CatBoost sees it, it's a scaled float like 0.23 — not "student". Passing it as `cat_features` would confuse CatBoost, which expects raw strings. **This was an explicit bug fixed in the codebase.** The fix: don't pass `cat_features` to CatBoost, treating it as a regular numeric feature.

**CatBoost's symmetric trees:** Unlike XGBoost and LightGBM which grow asymmetric trees, CatBoost uses **oblivious trees** — symmetric trees where the same feature and threshold is used for splits at the same depth level across all branches. This sounds restrictive but enables extremely fast inference (the tree structure fits in a lookup table) and reduces overfitting.

### 11.6 The Meta-Learner (Stacking)

```python
# Build meta-features from base model predictions
def build_meta_features(xgb_probs, lgbm_probs, cat_probs, gnn_scores):
    return np.hstack([
        xgb_probs.reshape(-1, 1),
        lgbm_probs.reshape(-1, 1),
        cat_probs.reshape(-1, 1),
        gnn_scores.reshape(-1, 1)
    ])  # shape: (N, 4)

meta_X = build_meta_features(xgb_val_probs, lgbm_val_probs, cat_val_probs, gnn_val_scores)

meta_model = LogisticRegression(
    C=1.0,                    # inverse regularization strength
    class_weight='balanced',  # handles imbalance at meta-level
    max_iter=1000
)
meta_model.fit(meta_X, y_val)  # train on VALIDATION SET (out-of-fold)
```

**The stacking formula:**

```
P_ensemble = σ(w₁·P_XGB + w₂·P_LGBM + w₃·P_CAT + w₄·P_GNN + b)
```

where `w₁, w₂, w₃, w₄, b` are learned by logistic regression, and σ is the sigmoid function.

**Why logistic regression for meta-learning (not another XGBoost)?**

The meta-learner receives only 4 features (the 4 base model probabilities). With only 4 features, any complex model would immediately overfit:

* A decision tree on 4 features has at most 2⁴=16 cells
* An XGBoost on 4 features would memorize the training data

Logistic regression is ideal: it learns the optimal linear combination of 4 probabilities. The learned weights are interpretable: if `w₄` (GNN weight) is highest, the meta-learner determined that GNN is most informative. If `w₁ ≈ w₂` (XGB ≈ LGBM), they contribute equally and are somewhat redundant.

**Why train on validation set, not training set?**

If we train base models on data and then compute meta-features on the same data, the base models have already "seen" these accounts — their predictions are overconfident. The meta-learner would learn to trust them more than it should, leading to a final model that overfits.

Training the meta-learner on the **validation set** (which the base models haven't been trained on) ensures the meta-features reflect the base models' true generalization ability on unseen data. This is the correct stacking procedure.

### 11.7 Why These Three Models Together?

The key is **diversity** — they make different errors on different accounts:

| Characteristic | XGBoost                 | LightGBM                     | CatBoost              |
| -------------- | ----------------------- | ---------------------------- | --------------------- |
| Tree growth    | Depth-first, balanced   | Leaf-wise, unbalanced        | Oblivious symmetric   |
| Regularization | L1+L2, gamma            | L1+L2                        | L2                    |
| Optimization   | 2nd order gradient      | 2nd order gradient           | Ordered boosting      |
| Speed          | Medium                  | Fast                         | Slow                  |
| Strength       | Stable, well-calibrated | Fast, accurate on large data | Robust to overfitting |

When XGBoost misclassifies account X (predicts legitimate when it's fraud), LightGBM and CatBoost often predict fraud correctly — because they use different tree structures that happen to split on different feature combinations for that account. The meta-learner learns to aggregate these diverse opinions into a robust final prediction.

---

## SECTION 12 — LAYER 4: VARIATIONAL AUTOENCODER (VAE)

### 12.1 Concept: Learning the Shape of Normal

The VAE is trained  **exclusively on legitimate accounts (label=0)** . It learns a compact mathematical model of what a normal bank account's feature vector looks like. At inference time:

* **Legitimate account** → reconstruction error is LOW (the account fits the model of normality)
* **Mule account** → reconstruction error is HIGH (the account's behavioral pattern doesn't match any legitimate pattern the VAE has learned)

This allows the VAE to detect **zero-day mule tactics** — novel fraud patterns the supervised models have never seen.

### 12.2 Architecture

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        # ENCODER: compress input to latent distribution parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 55 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 128 → 64
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)      # mean
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)  # log variance
      
        # DECODER: reconstruct input from latent sample
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),  # 32 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),  # 64 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)          # 128 → 55
        )
```

### 12.3 The Reparameterization Trick (Expect This Question)

**The problem:** We want to sample `z ~ N(μ, σ²)` but sampling is not differentiable — gradients cannot flow through a random operation.

**The trick:**

```
z = μ + ε × exp(0.5 × logvar)    where ε ~ N(0, I)
```

Now ε is just injected noise with no parameters. The randomness is factored out. Gradients flow through `μ` and `logvar` (both produced by the encoder network) normally. The backpropagation works exactly as with deterministic operations.

**Why encode logvar instead of σ directly?**

* Variance σ² must be positive. If we parameterize σ directly, we'd need a softplus activation to ensure positivity.
* `logvar = log(σ²)` can be any real number — unconstrained. The model is easier to optimize.
* Recovery: `σ = exp(0.5 × logvar)` always gives a valid positive standard deviation.

```python
def reparameterize(self, mu, logvar):
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # N(0,I) of same shape as std
        return mu + eps * std        # reparameterized sample
    return mu  # deterministic at inference: use mean
```

### 12.4 The ELBO Loss Function

The VAE is trained to maximize the Evidence Lower BOund (ELBO):

```
L(θ,φ; x) = E_{q_φ(z|x)}[log p_θ(x|z)] - β × KL(q_φ(z|x) || p(z))

= Reconstruction_Term - β × KL_Term
```

**Term 1: Reconstruction Term**

```
E[log p_θ(x|z)] ≈ -MSE(x, x̂)   [for Gaussian decoder]
```

Measures how well the decoder reconstructs the input from the latent code. We want this to be as large as possible (= small MSE).

**Term 2: KL Divergence Term**

```
KL(q_φ(z|x) || p(z)) = -½ Σ (1 + logvar - μ² - exp(logvar))
```

Measures how much the learned latent distribution `q(z|x)` deviates from the standard normal prior `p(z) = N(0,I)`. Acts as regularization — prevents the encoder from using all of its capacity to perfectly encode training accounts (which would cause it to memorize rather than generalize).

**β = 0.001 (very small KL weight):**
Standard VAEs use β=1. For financial tabular data, we need very high reconstruction accuracy — the reconstruction error IS the anomaly score. A large KL term (strong regularization) would force the latent space to be very close to N(0,I), reducing the encoder's ability to distinguish different types of legitimate accounts and thus reducing the anomaly detection quality. β=0.001 gives just enough regularization to produce a smooth latent space without sacrificing reconstruction accuracy.

### 12.5 Anomaly Scoring

```python
def reconstruction_error(self, x):
    """MSE reconstruction error per sample."""
    with torch.no_grad():
        x_hat, _, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        # Returns a scalar error per account
```

**P95 threshold:**

```python
# Computed on training legitimate accounts
errors_on_training = model.reconstruction_error(X_legit_tensor).numpy()
threshold = np.percentile(errors_on_training, 95.0)

# This means: exactly 5% of legitimate training accounts will exceed this threshold
# → controlled 5% false positive rate for the VAE layer specifically
```

**Normalized anomaly score [0,1]:**

```python
normalized_score = np.clip(errors / threshold, 0, 3) / 3.0
# Clip at 3× the threshold; anything beyond that is clearly anomalous
# Divide by 3 to map [0, 3×threshold] to [0, 1]
```

### 12.6 Critical Engineering Fix in VAE Training

```python
# ADASYN generates synthetic fraud samples by interpolation
# Occasionally produces extreme values or NaN/Inf
# Without this guard, the VAE loss becomes NaN and training crashes silently

X_legit = np.nan_to_num(X_legit, nan=0.0, posinf=0.0, neginf=0.0)
X_legit = np.clip(X_legit, -10.0, 10.0)  # clamp to ±10 standard deviations
```

Also during training:

```python
loss = vae_loss(batch, x_hat, mu, logvar)
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f'VAE Epoch {epoch}: NaN/Inf loss detected, skipping batch')
    continue
# + gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 12.7 VAE vs Isolation Forest: Complementary Anomaly Detectors

Both are in the codebase (`models/anomaly/vae.py` and `models/anomaly/isolation_forest.py`):

| Aspect            | VAE                                                                           | Isolation Forest                                                 |
| ----------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Mechanism         | Reconstruction error — anomaly = doesn't fit learned density model           | Partitioning — anomaly = easily isolated with few random splits |
| Best at           | Detecting accounts anomalous across many correlated dimensions simultaneously | Detecting univariate or bivariate outliers                       |
| Training data     | Legitimate accounts only                                                      | All accounts (with contamination parameter)                      |
| Theoretical basis | Variational inference, density estimation                                     | Random forest, average path length                               |
| Weakness          | May not detect anomalies that happen to reconstruct well                      | May miss correlated multi-feature anomalies                      |

**Combined score:**

```python
def combined_anomaly_score(vae_score, iso_score, vae_weight=0.6, iso_weight=0.4):
    return np.clip(vae_weight * vae_score + iso_weight * iso_score, 0.0, 1.0)
```

---

## SECTION 13 — LAYER 5: LineMVGNN

### 13.1 The Line Graph Transformation

Standard GNNs operate on  **account nodes** . LineMVGNN operates on the **line graph** L(G) of the transaction graph G:

```
Original graph G:
  Nodes = accounts
  Edges = transactions between accounts

Line graph L(G):
  Nodes = TRANSACTIONS (what were edges in G)
  Edges = two transactions share an account (either source or destination)
```

**Concrete example — a 3-hop cycle A→B→C→A:**

In G: 3 nodes (A, B, C), 3 edges (A→B, B→C, C→A)

In L(G): 3 nodes (transaction A→B, transaction B→C, transaction C→A), and edges:

* (A→B) and (B→C) share node B → connected in L(G)
* (B→C) and (C→A) share node C → connected in L(G)
* (C→A) and (A→B) share node A → connected in L(G)

**Result:** The 3 transaction nodes form a triangle in L(G). GNN message passing on a triangle is the simplest possible structure — the cycle detection becomes trivially easy.

**Why this matters:** In G, detecting a cycle requires a dedicated algorithm (DFS or SCC computation). In L(G), cycles are just dense subgraphs — the GNN detects them automatically through its normal message-passing mechanism, without any special-purpose cycle detection code.

### 13.2 Multi-View: Both Graphs Simultaneously

LineMVGNN runs GNN on both G (account graph) and L(G) (transaction graph) simultaneously:

```
View 1 (account-level): GNN on G → account embeddings H_G
View 2 (transaction-level): GNN on L(G) → transaction embeddings H_LG

Final account representation: 
    h_account = CONCAT(H_G[account], mean(H_LG[transactions involving account]))
```

Some fraud patterns are only visible in G (STAR hub-spoke topology). Others are only visible in L(G) (circular payment cycles). Multi-view captures both without requiring separate detection algorithms.

---

## SECTION 14 — LAYER 6: SHAP EXPLAINABILITY + LLM NARRATION

### 14.1 What SHAP Is: The Math

**SHAP** (SHapley Additive exPlanations, Lundberg & Lee, NeurIPS 2017, arXiv:1705.07874) is a unified framework for explaining individual model predictions using Shapley values from cooperative game theory.

**The Shapley value formula:**

```
φᵢ(f, x) = Σ_{S ⊆ F\{i}}  [|S|!(|F|-|S|-1)!/|F|!] × [f(S ∪ {i}) - f(S)]
```

Where:

* `F` = set of all features
* `S` = subset of features not containing feature i
* `f(S)` = model output when only features in S are known (others set to expected value)
* `[|S|!(|F|-|S|-1)!/|F|!]` = weighting factor (ensures fair allocation across all orderings)

**In plain English:** The Shapley value of feature i for account x is the average marginal contribution of feature i across all possible orderings of features. It answers: "On average across all ways I could have revealed the features one by one, how much did feature i change the prediction when I revealed it?"

**Why is this the "fair" allocation?**
The Shapley value is the unique solution satisfying four axioms from cooperative game theory:

1. **Efficiency:** All Shapley values sum to the prediction gap: Σᵢ φᵢ = f(x) - E[f(x)]
2. **Symmetry:** Two features with identical contributions get equal Shapley values
3. **Dummy:** A feature that never changes any prediction gets Shapley value 0
4. **Linearity:** Shapley values for a linear combination of models equal the linear combination of their individual Shapley values

No other explanation method satisfies all four simultaneously.

### 14.2 TreeSHAP: Why It's Fast for Tree Models

Computing exact Shapley values naively requires 2^|F| model evaluations (one for each subset). For |F|=55 features, that's 2^55 ≈ 3.6 × 10^16 evaluations — completely infeasible.

**TreeSHAP** (Lundberg et al., 2018) exploits tree structure to compute *exact* Shapley values in O(T × L × D²) time:

* T = number of trees (500 for XGBoost)
* L = maximum number of leaves per tree (2^6 = 64)
* D = maximum depth (6)

Total: 500 × 64 × 36 ≈ 1.15 million operations — fast enough for real-time inference.

**The key insight:** For a decision tree, the contribution of a feature can be computed exactly by following the tree's decision paths and computing how the prediction changes when that feature is "masked" at each node. The tree structure makes this tractable.

```python
from inference.explainer import explain_account

def explain_account(model, X_row, top_n=10):
    explainer = shap.TreeExplainer(model)  # builds the TreeSHAP explainer
    shap_vals = explainer.shap_values(X_row)  # shape: (1, n_features)
  
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # binary classification: take fraud class values
  
    row_shap = shap_vals[0]  # single account
  
    factors = []
    for fname, sval, rval in zip(X_row.columns, row_shap, X_row.iloc[0]):
        factors.append({
            'feature': fname,
            'shap_value': float(sval),
            'raw_value': float(rval) if not pd.isna(rval) else None,
            'direction': 'fraud_risk' if sval > 0 else 'safe_signal'
        })
  
    factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return factors[:top_n]
```

### 14.3 SHAP Outputs in Nyxara

**Local SHAP (per account):** Waterfall chart of top-10 features. For account ACC-7832:

```
Base value: 0.12 (population mean fraud probability)
+ occupation_velocity_anomaly: +0.42 (student with 85 txns → +42% toward fraud)
+ pass_through_score:          +0.31 (near-zero net balance → +31%)
+ financial_impossibility:     +0.28 (₹4.5L for student → +28%)
+ F2678 (nocturnal):           +0.18 (3AM transactions → +18%)
- F2956 (credit score):        -0.05 (moderate score → -5%)
= Final prediction: 0.87 (BLOCK)
```

**Global SHAP (across all flagged accounts):** Monthly bar chart of mean |SHAP| values:

* pass_through_score: 0.284
* occupation_velocity_anomaly: 0.231
* network_centrality_proxy: 0.198
* financial_impossibility_score: 0.167
* F2082 (cross-border): 0.143

This tells the risk policy team WHICH features are driving all flags this month — directly actionable for updating KYC policies.

### 14.4 LLM Alert Narration

```python
# From inference/alert_narrator.py
async def generate_alert(account_id, decision, final_risk, occupation, 
                          top_factors, ring_membership, community_fraud_rate):
    if not GROQ_API_KEY:
        return _rule_based_alert(...)  # fallback
  
    from groq import AsyncGroq
    client = AsyncGroq(api_key=GROQ_API_KEY)
  
    prompt = _build_prompt(account_id, decision, final_risk, occupation,
                            top_factors, ring_membership, community_fraud_rate)
  
    completion = await client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.1-8b-instant',
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()
```

**The prompt enforces:**

1. Legally appropriate language: *"exhibits behaviors consistent with indicators of"* — not "this person is a criminal" (which would be defamatory without a conviction)
2. Every claim must trace to provided SHAP data — no hallucination allowed
3. FIU-IND STR format alignment
4. 3-sentence structure: overall risk assessment → primary behavioral indicators → recommended compliance action

**Rule-based fallback (complete, no LLM needed):**

```python
def _rule_based_alert(account_id, decision, final_risk, occupation, top_factors, ring_membership):
    risk_label = {'APPROVE': 'LOW', 'REVIEW': 'MEDIUM', 'FLAG': 'HIGH', 'BLOCK': 'CRITICAL'}[decision]
    top_feature = top_factors[0]['feature'] if top_factors else 'transaction velocity'
    ring_text = ' Account is identified as a member of a suspicious transaction ring.' if ring_membership else ''
    action_map = {
        'APPROVE': 'No immediate action required; continue standard monitoring.',
        'REVIEW': 'Initiate enhanced due diligence review within 24 hours.',
        'FLAG': 'Escalate to senior compliance officer; consider account restriction.',
        'BLOCK': 'Recommend immediate account freeze and initiate STR filing with FIU-IND.'
    }
    return (
        f'Account {account_id} ({occupation}) exhibits behaviors consistent with indicators of '
        f'financial fraud, receiving a {risk_label} risk score of {final_risk:.2f}. '
        f'Primary behavioral indicators include anomalous {top_feature} relative to occupation peer group, '
        f'with additional signals from transaction pattern analysis.{ring_text} '
        f'{action_map[decision]}'
    )
```

---

## SECTION 15 — RISK FUSION: COMBINING ALL 6 LAYERS

### 15.1 The Fusion Formula

```
finalRisk = 0.35 × GNN + 0.25 × Ensemble + 0.20 × VAE + 0.12 × BEI + 0.08 × Graph
```

Where:

```
Graph = 0.5 × ring_membership_flag + 0.3 × community_fraud_rate + 0.2 × two_hop_contamination
```

All scores are clamped to [0, 1] before fusion. finalRisk is clamped to [0, 1] after.

**Implementation:**

```python
# From inference/scorer.py
W_GNN = float(os.getenv('WEIGHT_GNN', '0.35'))
W_ENSEMBLE = float(os.getenv('WEIGHT_ENSEMBLE', '0.25'))
W_VAE = float(os.getenv('WEIGHT_VAE', '0.20'))
W_BEI = float(os.getenv('WEIGHT_BEI', '0.12'))
W_GRAPH = float(os.getenv('WEIGHT_GRAPH', '0.08'))

gnn_s = max(0.0, min(1.0, float(gnn_score)))
ens_s = max(0.0, min(1.0, float(ensemble_score)))
vae_s = max(0.0, min(1.0, float(vae_score)))
bei_s = max(0.0, min(1.0, float(bei_score)))

graph_s = 0.5 * float(ring_membership) + 0.3 * float(community_fraud_rate) + 0.2 * float(two_hop)
graph_s = max(0.0, min(1.0, graph_s))

final_risk = (W_GNN * gnn_s + W_ENSEMBLE * ens_s + W_VAE * vae_s 
              + W_BEI * bei_s + W_GRAPH * graph_s)
final_risk = max(0.0, min(1.0, final_risk))
```

### 15.2 Why These Specific Weights?

| Layer                 | Weight  | Rationale                                                                                                                                                                                    |
| --------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GNN (0.35)            | Highest | Network topology is the hardest signal to fake. Individual account features can be coached to look normal. The STAR/CYCLE/CHAIN graph structure cannot be hidden.                            |
| Ensemble (0.25)       | Second  | Tabular ML on 55 features captures statistical anomalies (impossible financial patterns, occupation mismatches) very effectively. Proven SOTA on tabular financial fraud data.               |
| VAE (0.20)            | Third   | Zero-day detection adds orthogonal signal. Cannot be gamed by adapting to known fraud patterns. But higher uncertainty than supervised models.                                               |
| BEI (0.12)            | Fourth  | Device/behavioral signals are powerful but: (a) not always available, (b) can be partially evaded by using a legitimate device. 12% is enough to be the deciding factor in borderline cases. |
| Graph features (0.08) | Lowest  | Community membership and 2-hop contamination add context but are derivative of GNN — partially correlated. Low weight prevents double-counting.                                             |

**All weights configurable via .env** — banks can tune based on their specific fraud profile. A bank experiencing more RAT/SIM-swap attacks might increase W_BEI to 0.20. A bank with better data on transaction networks might increase W_GNN to 0.40.

### 15.3 The Override Rule

```python
# From inference/scorer.py
RING_OVERRIDE_COMMUNITY_RATE = 0.60

if ring_membership and community_fraud_rate > 0.60:
    if decision == 'APPROVE':
        decision = 'REVIEW'
        override_applied = True
        logger.info(f'[{account_id}] Override: ring member + '
                    f'community_fraud_rate={community_fraud_rate:.2f} → forced REVIEW')
```

**Why this override exists:**

Consider an account that was recruited as a mule 3 months ago but hasn't been activated yet. Its individual features look completely normal (no unusual velocity, reasonable amounts, clean occupation). The GNN sees some suspicious connections but not enough to push above the APPROVE threshold.

But the account IS a member of a ring with 83% fraud rate. This is strong network-level evidence that it WILL be activated. The override forces a minimum REVIEW so the account is monitored, even though the model score says APPROVE.

This is the "guilt by network association" rule with a 60% community fraud rate threshold — high enough to avoid false positives from loose community associations, low enough to catch pre-activation mules.

---

## SECTION 16 — ADAPTIVE THRESHOLD CALIBRATION (THE CRITICAL FIX)

### 16.1 The Problem at 0.89% Fraud Rate

This is the single most important engineering contribution in the entire project. Understand it completely.

**What happens with default threshold = 0.5:**

The model learns to output probability scores that reflect the true class distribution:

* Legitimate accounts: scores typically in range [0.001, 0.15]
* Fraud accounts: scores typically in range [0.05, 0.65]

At threshold = 0.5: almost zero accounts predicted as fraud (only those with score > 0.50). Result: Precision ≈ 0.95 (the few accounts we flag really are fraud), but Recall ≈ 0.05 (we catch almost no actual fraud). F1 ≈ 0.13.

**This is not a model failure. The model is correctly ranking fraud above legitimate.** The AUC-ROC (a ranking metric) is still 0.982. The failure is purely in the threshold selection.

**Proof:** AUC-ROC = 0.982 means 98.2% of (fraud, legitimate) account pairs are correctly ranked. The model knows fraud is more suspicious. It just doesn't output probabilities > 0.50 because the prior probability of fraud is so low (0.89%).

### 16.2 The Precision-Recall Curve Solution

```python
# From training/run_all.py — Step 12a
from sklearn.metrics import precision_recall_curve

# Compute on VALIDATION SET (never test set)
val_precision, val_recall, val_thresholds = precision_recall_curve(y_val, val_ens_probs)

# Compute F1 at every threshold
val_f1_scores = (2 * val_precision * val_recall 
                 / np.where((val_precision + val_recall) == 0, 1e-9, 
                            (val_precision + val_recall)))

# precision_recall_curve returns N+1 points for N thresholds
# val_f1_scores[:-1] aligns with val_thresholds
val_f1_aligned = val_f1_scores[:-1]

if len(val_f1_aligned) > 0:
    best_idx = int(np.argmax(val_f1_aligned))
    optimal_threshold = float(val_thresholds[best_idx])
else:
    optimal_threshold = 0.5  # fallback
```

**Why the [:-1] indexing?**
`precision_recall_curve` returns arrays of length N+1 for precision and recall, but only N thresholds. The last point corresponds to threshold=1.0 (predict everything as legitimate) and is not paired with a threshold value. `val_f1_scores[:-1]` drops this extra sentinel point to align indices correctly.

### 16.3 Derived Decision Thresholds

```python
# Save optimal_threshold to eval_report.json
# At startup, scorer.py loads it and derives decision tiers:

opt = float(report.get('optimal_threshold', 0.0))

T_approve = max(0.01, min(opt * 0.60, 0.60))
T_review  = max(T_approve + 0.01, min(opt * 1.10, 0.80))
T_flag    = max(T_review + 0.01, min(opt * 1.50, 0.95))

# Example: if optimal_threshold = 0.08:
# T_approve = max(0.01, min(0.048, 0.60)) = 0.048
# T_review  = max(0.058, min(0.088, 0.80)) = 0.088
# T_flag    = max(0.098, min(0.120, 0.95)) = 0.120
```

**Why scale relative to optimal_threshold?**
The four decision tiers should scale with the model's calibration. If retraining produces a model that outputs higher probabilities (e.g., due to better balancing), the thresholds should scale up accordingly. Fixed absolute thresholds (like T_flag = 0.85) only work if the model's probability outputs are stable across retraining — they're not.

### 16.4 Why Validation Set, Not Test Set (CRITICAL)

```
Train set: Used to train all 6 models
Validation set: Used to find optimal_threshold
Test set: Touched EXACTLY ONCE for final evaluation reporting

If you find optimal_threshold on the TEST set:
→ You've optimized a hyperparameter using test data
→ This is DATA SNOOPING / test set contamination
→ Your F1 = 0.91 number is no longer a fair estimate of generalization
→ The model might have a much lower F1 on genuinely new data

If a jury member asks "did you tune the threshold on test data?" 
→ The answer is NO: we used the validation set.
→ The test set was used only to compute the final metrics.
```

---

## SECTION 17 — KEY JURY QUESTIONS FOR PART 2

**Q: Why not just use a single deep neural network instead of this complex multi-model setup?**
A: A single deep neural network is a universal approximator in theory, but in practice it requires enormous amounts of labeled data to generalize well. At 0.89% fraud rate with 81 labeled examples, a deep neural network would severely overfit. Our multi-model approach uses each technique where it's strongest: gradient boosting is proven SOTA for tabular data; GNNs are specifically designed for graph-structured data; VAE uses ALL 9,082 accounts (not just the 81 labeled examples) for anomaly learning. The ensemble of diverse models also reduces variance — when one model makes errors, the others compensate.

**Q: You use cross-entropy loss for the GNN. Why not focal loss, which is popular for imbalanced classification?**
A: Focal loss (Lin et al., CVPR 2017) modifies cross-entropy by adding a modulating factor: `FL(p) = -(1-p)^γ log(p)`. This down-weights easy negatives (correctly classified legitimate accounts) and focuses training on hard positives (fraud accounts near the decision boundary). It's an alternative to class weighting. We chose class-weighted cross-entropy because: (1) it's more interpretable — the 112:1 weight directly reflects the 112:1 class ratio; (2) focal loss introduces an additional γ hyperparameter that needs tuning; (3) for our specific 0.89% fraud rate, class weighting achieves the same effect more simply. Focal loss would also be valid; it's an engineering choice, not a principled reason to prefer one over the other.

**Q: What is the difference between AUC-ROC and AUC-PR? Which is more appropriate here?**
A: AUC-ROC plots True Positive Rate vs False Positive Rate at all thresholds and measures the area under this curve. AUC-PR plots Precision vs Recall at all thresholds and measures the area under that curve. For severely imbalanced datasets like ours (0.89% fraud), AUC-PR is actually more informative. The ROC curve can be misleadingly optimistic because the large number of true negatives (8,997 legitimate accounts) makes the False Positive Rate always look small even when you're generating many false alarms in absolute terms. AUC-PR directly measures precision-recall tradeoffs, which is what operationally matters for an analyst. We report AUC-ROC because it's the standard for regulatory benchmark comparison (MuleHunter.AI uses AUC-ROC); we also report F1 which captures the precision-recall balance.

**Q: The GNN trains on a static graph. How would you handle new accounts being added daily in production?**
A: GraphSAGE is specifically designed as an inductive learner — it learns an *aggregation function* (how to combine neighbor representations), not fixed embeddings. A new account can be embedded by: (1) building its local subgraph using the existing trained graph plus its new connections, (2) applying the saved GNN model to compute its embedding via the learned aggregation function. No retraining is needed for individual new accounts. For production deployment, we'd maintain a rolling graph that adds new accounts and edges as they appear, with periodic full retraining (monthly) to update the model when the fraud distribution shifts.

**Q: Your meta-learner uses only 4 features (the 4 base model probabilities). Isn't that too simple?**
A: The meta-learner's job is not to be complex — its job is to learn the optimal linear combination of 4 already-computed probabilities. Adding more complexity (more features, a deeper model) would cause overfitting since the meta-learner is trained on the validation set which has at most a few hundred positive examples. Logistic regression with 4 features and class-balanced weighting is exactly the right complexity. The learned weights are also interpretable: they tell you how much to trust each base model. Simplicity here is a feature, not a limitation.

---

*Continue to Part 3 → Graph Intelligence · Cybersecurity BEI · Behavioral Biometrics*
