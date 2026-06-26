# Nyxara Master Study Guide: PhD-Level Clarity for the Grand Finale

Welcome to the definitive guide for your Nyxara platform. This document contains every algorithm, formula, threshold, architecture detail, and data flow required to answer the most granular questions the jury can throw at you. Memorize these sections, and you will dominate the presentation.

---

## 1. System Architecture Overview
Nyxara is composed of five microservices communicating seamlessly:
1. **Frontend (Port 3000):** React 18, Vite, Three.js, D3.js, Recharts, Custom Canvas.
2. **Backend (Port 8080):** Node.js, Express, Socket.IO, MongoDB. Orchestrates the pipeline.
3. **AI Engine (Port 8001):** FastAPI, PyTorch, XGBoost, LightGBM, CatBoost, NetworkX.
4. **Cybersec Engine (Port 8002):** FastAPI, Redis. Handles BEI and device fingerprinting.
5. **Blockchain Audit (Port 8003):** FastAPI, Motor (Async Mongo). Handles Merkle tree hashing.

---

## 2. Dataset Engineering & Preprocessing Pipeline
The AI Engine pipeline reduces 3,924 raw features to ~50 power features.

### The 5-Stage Feature Selection
1. **Variance Threshold:** Drops features with variance < 0.01 (removes near-constant noise). Reduces 3,924 → ~1,800.
2. **Missing Analysis:** Drops columns with >50% missing values. Creates `{col}_missing` flags for 40-50% missing. Crucially, always creates `F3043_missing` (account tenure), as missing tenure indicates a ghost account. Reduces ~1,800 → ~1,200.
3. **Mutual Information (MI) Ranking:** Captures non-linear dependencies. Keeps top 150 + 18 `BANK_KEY_FEATURES` (force-included). Reduces ~1,200 → ~150.
4. **Correlation Deduplication:** Drops features with Pearson |r| > 0.95 to prevent multicollinearity, keeping the one with higher MI. Reduces ~150 → ~100.
5. **SHAP Forward Selection:** Trains a quick XGBoost (100 trees), ranks by `mean(|SHAP|)`, keeps top 50 + bank key features. Reduces ~100 → 50.

### 12 Composite Features (Engineered BEFORE encoding)
These are critical domain-specific features engineered from raw data:
1. **`occupation_velocity_anomaly`**: `F3894 / 95th_percentile(F3894 | occupation)`. Detects velocity anomalies relative to peers.
2. **`pass_through_score`**: `F527 × |F2737|`. High in+out flow correlation.
3. **`network_centrality_proxy`**: `F531 × log(1 + F1692)`.
4. **`dormancy_activation_signal`**: `F3043_missing × F3894`. Dormant account suddenly active.
5. **`financial_impossibility_score`**: `|F3836| / income_proxy`. Amount impossibly large for occupation.
6. **`peer_deviation_combined`**: `|F2582| + |F2678|`.
7. **`structuring_risk`**: `F2122 × F3894`. Splitting transactions to avoid thresholds.
8. **`cross_border_occupation_risk`**: `F2082 × risk_weight[occupation]`.
9. **`temporal_burst_index`**: `F3894 / max(F3043, 1)`. Sudden burst relative to age.
10. **`international_flag_occupation`**: Binary flag if cross-border + high-risk occupation (student, housewife, retired).
*(Note: risk_cluster_membership and two_hop_contamination are graph placeholders filled later).*

### Encoding & Scaling
- **Categoricals (`F3889`, `F3891`)** are `LabelEncoded` to `int32`.
- **Numerics** are scaled using `RobustScaler` `(x - median) / IQR` to resist financial outliers. Encoded columns are skipped.

### Class Imbalance Handling
Fraud rate is typically < 1% (0.89% in training).
- Uses **ADASYN** (`sampling_strategy=0.3`) to generate synthetic samples in difficult regions.
- Gradient boosting uses `scale_pos_weight = n_negative / n_positive`.
- Meta-learner uses `class_weight='balanced'`.

---

## 3. The 6-Layer AI Detection Stack

### Layer 1: Temporal GraphSAGE + GAT (Graph Neural Network)
A 3-layer hybrid PyTorch architecture that learns network topologies.
- **Edges:** Built using KNN (Cosine dist ≤ 0.3), Same Branch (`F3889`), and Shared Device (`F1692 > 5`).
- **Layer 1 (GraphSAGE):** `SAGEConv(in → 128)` + BatchNorm + ReLU + Dropout(0.3).
- **Layer 2 (GAT):** `GATConv(128 → 32 × 4 heads = 128)` + BatchNorm + ReLU. Learns hub-spoke attention weights.
- **Layer 3 (GraphSAGE + Residual):** `SAGEConv(128 → 128)` + residual skip connection `h = ReLU(h + skip_proj(x))` to prevent over-smoothing.
- **Classifier Head:** Linear(128→64) → ReLU → Linear(64→2).
- **Training:** 450 epochs, Adam (lr=0.001), ReduceLROnPlateau, Early Stopping (patience=30) based on AUC.

### Layer 2: Graph-Temporal Contrastive Transformer (GTCT)
*(Part of the extended architecture for zero-day mules)* Uses SimCLR-style contrastive loss to maximize similarity between augmented views of the same subgraph and minimize against others. Temperature $\tau = 0.07$.

### Layer 3: Ridge-Stacked Tabular Ensemble
Three gradient boosting models, heavily optimized for CPU:
- **XGBoost:** 500 trees, depth 6, lr 0.05, `tree_method="hist"`.
- **LightGBM:** 500 trees, 63 leaves, min_child 20.
- **CatBoost:** 500 trees, depth 6, l2_leaf_reg=3. (Treats label-encoded ints as numeric).
- **Stacker:** Meta-learner is `LogisticRegression` taking probabilities from XGB, LGBM, CAT, and GNN.

### Layer 4: Variational Autoencoder (VAE)
Trained **exclusively on legitimate accounts**. Detects anomalies based on reconstruction error.
- **Architecture:** MLP Encoder (→ 64 → μ:32, logvar:32) → Reparameterization → MLP Decoder.
- **Loss:** $MSE(x, \hat{x}) - 0.001 \times D_{KL}$. Low KL weight prevents posterior collapse.
- **Threshold:** 95th percentile of training reconstruction errors. Anomaly score = `clip(error/threshold, 0, 3) / 3.0`.

### Layer 5: LineMVGNN Cycle Detection
Transforms graph into a Line Graph (nodes = transactions) to detect directed cycles ($A \rightarrow B \rightarrow C \rightarrow A$) spanning institutions.

### Layer 6: Explainability (SHAP & LLM)
- Computes `shap.TreeExplainer` on XGBoost model.
- Passes top factors to Groq `llama-3.1-8b-instant`. Prompt enforces FIU-IND STR formatting and bans hallucination.

---

## 4. Risk Fusion & Adaptive Thresholds

### Risk Fusion Formula
Final risk is a weighted sum (clamped to [0, 1]):
**`finalRisk = 0.35(GNN) + 0.25(Ensemble) + 0.20(VAE) + 0.12(BEI) + 0.08(Graph)`**

### Adaptive Thresholds
Instead of hardcoding, Nyxara finds the optimal threshold $\theta^*$ that maximizes the F1 score on the validation precision-recall curve.
- `T_APPROVE = max(0.01, min(θ* × 0.60, 0.60))`
- `T_REVIEW = max(T_APPROVE + 0.01, min(θ* × 1.10, 0.80))`
- `T_FLAG = max(T_REVIEW + 0.01, min(θ* × 1.50, 0.95))`

**Override Rule:** If an account is in a ring AND community fraud rate > 0.60, a decision of APPROVE is forced to **REVIEW**.

---

## 5. Kinetic Cyber-Security (BEI)

The Browser Environment Intelligence (BEI) score is the maximum of four risk vectors:

1. **Mouse Entropy:** 16 angular buckets. Shannon entropy max 4.0 bits. Normalized `1 - H/4.0`. Score < 0.3 indicates straight-line bot movement.
2. **Keystroke Dynamics:** Coefficient of Variation (CV = std/mean). Human CV is 0.3-0.8. Bot CV < 0.05 yields 0.95 consistency risk.
3. **Session Velocity:**
   - Events/sec > 50 → score 1.0.
   - Form fill < 3.0 seconds → `bot_fill_detected`, score ≥ 0.8.
   - *Special Case:* Bot mouse AND Bot form fill → automatic 0.95 BEI.
4. **JA3 Fingerprinting:** Hashes TLS ClientHello. Matches known malware (e.g., Trickbot: `e7d705a3286e...`) for instant block.
5. **Velocity Tracking (Redis):** Tracks accounts per device/IP using sets and TTLs. >10 accounts/device/24h is a hard block.

---

## 6. Graph Intelligence (Rings & Communities)

**Ring Detector (NetworkX, DFS-based):**
1. **STAR:** Node with out_degree ≥ 5 AND risk ≥ 0.5. Hub and mules.
2. **CHAIN:** DFS following nodes with exactly 1 successor. Max 6 hops. Source, relay, terminus.
3. **CYCLE:** Tarjan's Strongly Connected Components (size ≥ 3). Reciprocity drives confidence.
4. **CLUSTER:** Undirected weakly connected components with density ≥ 0.3.

**Community Detection (Louvain):**
Optimizes modularity. Each community gets a `community_fraud_rate` (average of members' GNN scores > 0.5). This propagates back to individuals as "guilt by association."

**PageRank:**
Finds hubs ($\alpha=0.85$). PR > 0.8 + out_degree ≥ 5 = Orchestrator.

---

## 7. Blockchain Audit & Compliance

**Merkle Tree:**
- Every 50 decisions, a batch is sealed.
- Leaf hash: `SHA256("account_id|risk_score|decision|timestamp")`.
- Tree is built bottom-up (duplicates last node if odd).
- Root stored in Mongo `audit_batches`.
- `verify_chain()` recomputes all leaves. Any database tampering breaks the root hash.

**STR Auto-Draft:**
Maps system flags to FATF codes:
- `ringMembership` → FATF-ML-01
- `F527 > 0.9` → FATF-ML-02 (Pass-through)
- `F2082 > 0` → FATF-ML-04 (International)
- `F3043 == null` → FATF-ML-06 (Ghost account)
- `F2122 > 0.4` → FATF-ML-08 (Structuring)

---

## 8. Frontend Deep Dive (React, Canvas, Three.js, D3)

Nyxara's frontend is a visual powerhouse combining multiple rendering engines:
- **`Analyzer.jsx` (1659 lines):** The crown jewel. Uses raw Canvas 2D for the Risk Dial, Mule DNA Radar chart, Biometrics HUD (sonar sweep, keystroke CV), and a manual 3D-to-2D projected graph. AI Co-pilot streams responses at 18ms intervals.
- **`NetworkGraph.jsx`:** **Three.js** implementation. Uses a custom force layout hook with spatial bucketing for performance. Features `InstancedMesh` for O(1) draw calls, SpriteMaterial halos, and node scaling based on PageRank roles.
- **`AccountNetworkGraph.jsx`:** **D3.js** implementation. Uses `forceSimulation`, `forceLink`, `forceManyBody`, `forceCollide`. Renders animated flow particles along transaction edges and SVG glow filters for risk levels.
- **`RiskAtlas.jsx`:** Map of India rendering state-level fraud density using SVG/Canvas.
- **WebSocket:** `AlertContext.jsx` listens to `ws://localhost:8001` for `new_alert` and `alert_updated` events, broadcasting in real-time to the dashboard.

---

## 9. Key Numbers to Memorize
- **Dataset:** 9,082 accounts, 3,924 raw features → ~50 final features.
- **Performance:** AUC-ROC **0.982**, F1 **0.910**, Precision 0.940, Recall 0.890.
- **Weights:** GNN(0.35) + Ens(0.25) + VAE(0.20) + BEI(0.12) + Graph(0.08) = 1.0.
- **Cost:** ₹0 infrastructure cost via open-source stack.

*Own these details, explain the architectures with confidence, and the jury will have no choice but to be impressed. Good luck!*
