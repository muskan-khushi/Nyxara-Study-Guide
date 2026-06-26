# 🔮 NYXARA — GRAND FINALE MASTERY GUIDE

## Part 3 of 4: Graph Intelligence · Ring Detection · Cybersecurity BEI · Behavioral Biometrics · Velocity Tracking · JA3

> **Where we are:** Part 1 covered problem domain, dataset, features. Part 2 covered the 6-layer AI stack and risk fusion. This part covers the graph intelligence subsystem (ring detection, Louvain communities, PageRank) and the entire cybersecurity engine (BEI, behavioral biometrics, JA3 fingerprinting, Redis velocity tracking). These are architecturally separate from the AI engine but contribute 12% + 8% = 20% of the final risk score and often provide the deciding signals in borderline cases.

---

## SECTION 18 — GRAPH INTELLIGENCE: RING DETECTION

### 18.1 Why Ring Detection Deserves Its Own System

The GNN (Layer 1) outputs a continuous fraud probability for each account based on its learned graph representation. Ring detection is different — it is a **structural classification** of the entire network pattern, not a per-account score. A ring detector asks: "Does this cluster of accounts form a recognizable money laundering topology?"

This matters because:

1. A hub account with risk score 0.72 might not cross the FLAG threshold alone
2. But if the ring detector confirms it is the hub of a 6-node STAR ring with fraud_rate = 0.91, the override rule forces REVIEW minimum
3. The ring classification also provides the structural narrative for the STR auto-draft ("Account is the HUB of a 6-node STAR ring")
4. Ring membership directly feeds the `risk_cluster_membership` composite feature and `graph_score` in the fusion formula

### 18.2 The Four Ring Typologies: Complete Reference

**File:** `ai-engine/models/gnn/ring_detector.py`

#### STAR — Hub-and-Spoke (Most Common, Highest Confidence)

```
        [HUB]
       / | | \ \
      M1 M2 M3 M4 M5
```

**What it is:** A central orchestrator account that receives money and immediately fans it out to multiple mule accounts ("spokes"). The hub typically has high transaction velocity while the spokes each handle smaller individual amounts.

**Detection criteria:**

```python
def _detect_star(G, node, account_ids, min_spokes=5):
    out_deg = G.out_degree(node)          # edges going OUT from this node
    total_deg = G.degree(node)            # total edges (in + out)
  
    if out_deg < min_spokes:              # must have enough outgoing connections
        return None
  
    spokes = list(G.successors(node))    # all accounts this node sends to
    if len(spokes) < min_spokes:
        return None
  
    hub_id = account_ids[node]
    members = [hub_id] + [account_ids[s] for s in spokes]
    roles = {hub_id: 'hub'} | {account_ids[s]: 'mule' for s in spokes}
  
    # Confidence increases with number of spokes
    confidence = min(0.40 + out_deg * 0.04, 0.99)
  
    return Ring(ring_id=f'STAR_{hub_id}', shape='STAR', accounts=members,
                roles=roles, hub_node=hub_id, ...)
```

**Why min_spokes=5 (not 3)?**
The KNN graph connects behaviorally similar accounts — a legitimate account with 3–4 similar-looking neighbors is plausible (same employer, same bank branch, same demographic). Five or more similar accounts all connected to the same hub is statistically much harder to explain by coincidence. The threshold of 5 is calibrated to give high confidence detections while minimizing false positives.

**Why GNN risk > 0.50 for hub candidates?**

```python
for node in G.nodes():
    risk = G.nodes[node].get('risk', 0.0)
    if risk < 0.50:          # skip low-risk nodes as hub candidates
        continue
    ring = _detect_star(G, node, account_ids, min_spokes=5)
```

The KNN graph connects ALL behaviorally similar accounts, including clusters of legitimate accounts with similar profiles (e.g., all students at the same university). Without the GNN risk filter, we'd flag every dense cluster as a STAR ring. The 0.50 GNN threshold ensures we only investigate nodes that the GNN already identifies as suspicious.

**Confidence formula:** `min(0.40 + out_degree × 0.04, 0.99)`

* 5 spokes: 0.40 + 0.20 = 0.60
* 10 spokes: 0.40 + 0.40 = 0.80
* 15 spokes: 0.40 + 0.60 = 1.00 → capped at 0.99

More spokes = higher confidence because the probability of legitimate accounts naturally forming a hub-spoke pattern decreases exponentially with the number of spokes.

---

#### CYCLE — Closed-Loop Circular Laundering

```
A → B → C → D → A
```

**What it is:** A closed cycle where money circulates through multiple accounts, making it impossible to trace the origin. Each account receives from one and sends to another in a circle. When the money exits the cycle (one account sends to an external cash-out point), it appears to have come from the previous account in the cycle rather than the original source.

**Detection method — Strongly Connected Components (SCC):**

An SCC in a directed graph is a maximal set of nodes where every node is reachable from every other node. A cycle A→B→C→A is an SCC of size 3.

```python
def _detect_cycles(G, account_ids):
    rings = []
    for scc in nx.strongly_connected_components(G):
        if len(scc) < 3:              # minimum 3 nodes
            continue
      
        members = [account_ids[n] for n in scc]
        sub = G.subgraph(scc)
        recip = nx.reciprocity(sub) if sub.number_of_edges() > 0 else 0.0
      
        roles = {m: 'cycler' for m in members}  # all accounts are equal in a cycle
      
        rings.append(Ring(
            ring_id = f'CYCLE_{"_".join(sorted(members)[:3])}',
            shape = 'CYCLE',
            confidence = min(0.50 + recip * 0.40, 0.99),
            ...
        ))
    return rings
```

**Reciprocity:** `nx.reciprocity(G) = (number of edges with a mutual counterpart) / (total edges)`

A cycle A→B→C→A has 3 edges but no mutual edges (A→B but not B→A). Reciprocity = 0 for a pure directed cycle. However, in real mule rings there are often partial reciprocal transfers (cycler accounts also send money back sometimes), so reciprocity > 0 indicates a more complex, real-world cycle. The confidence formula `0.50 + recip × 0.40` gives:

* Pure directed cycle (recip=0): confidence = 0.50
* Fully mutual cycle (recip=1): confidence = 0.90

**Why SCCs for cycle detection?** The Kosaraju or Tarjan SCC algorithm runs in O(V+E) time — extremely efficient. It finds ALL cycles simultaneously, including complex multi-nested cycles. A dedicated DFS-based cycle finder would be O(V+E) per starting node — much slower.

---

#### CHAIN — Sequential Relay Laundering

```
[SOURCE] → [RELAY1] → [RELAY2] → [RELAY3] → [TERMINUS]
```

**What it is:** A linear sequence where money passes through a series of "relay" accounts before reaching the final cash-out point (terminus). Each relay receives from one account and immediately sends to the next. This creates distance between the crime and the cash-out.

**Detection method — DFS from in_degree=0 nodes:**

```python
def _detect_chain(G, start, account_ids):
    path = [start]
    visited = {start}
    current = start
    deadline = time.time() + TIME_BUDGET  # 25 seconds per chain search
  
    while time.time() < deadline and len(path) < MAX_HOPS:  # MAX_HOPS = 6
        succs = [s for s in G.successors(current) if s not in visited]
        if len(succs) != 1:   # chain node must have exactly 1 outgoing connection
            break
        nxt = succs[0]
        if G.in_degree(nxt) != 1:  # chain node must have exactly 1 incoming connection
            break
        path.append(nxt)
        visited.add(nxt)
        current = nxt
  
    if len(path) < 3:  # minimum 3 nodes to be a meaningful chain
        return None
  
    members = [account_ids[n] for n in path]
    roles = {members[0]: 'source', members[-1]: 'terminus'}
    roles.update({m: 'relay' for m in members[1:-1]})
  
    return Ring(
        ring_id = f'CHAIN_{members[0]}_{members[-1]}',
        shape = 'CHAIN',
        confidence = min(0.4 + len(path) * 0.06, 0.95),
        ...
    )
```

**The in_degree=1 and out_degree=1 constraints:** A true relay account in a chain has exactly one account sending to it and sends to exactly one account. Any deviation (multiple senders or multiple receivers) means it's not a pure relay — it's a hub or bridge. These constraints also prevent the DFS from following into complex subgraphs.

**Why start from in_degree=0 nodes?**
The SOURCE of a chain has no incoming connections — it's where money enters the laundering chain. Starting DFS from in_degree=0 nodes ensures we find the full chain from beginning to end, not starting from the middle.

**MAX_HOPS=6 and 25-second budget:** Prevents the DFS from exploring extremely long paths that are likely false positives (legitimate long transaction chains). The time budget is a safety valve for large, densely connected graphs.

**Confidence formula:** `min(0.4 + len(path) × 0.06, 0.95)`

* 3-hop chain: 0.4 + 0.18 = 0.58
* 6-hop chain: 0.4 + 0.36 = 0.76

Longer chains are more confident because legitimate accounts rarely form long strict linear chains.

---

#### CLUSTER — Dense Mutual Transaction Subgraph

```
  A ←→ B
  ↕    ↕
  C ←→ D
```

**What it is:** A group of accounts that all transact heavily with each other, forming a dense subgraph. Unlike the other topologies, there's no clear hub or direction — all accounts are roughly equal participants. Common in situations where a criminal organization runs a "washing pool" where money circulates within the group.

**Detection method — Louvain density on weakly connected components:**

```python
def _detect_clusters(G, account_ids):
    rings = []
    undirected = G.to_undirected()  # direction doesn't matter for cluster density
  
    for component in nx.connected_components(undirected):
        if len(component) < 3:
            continue
      
        sub = undirected.subgraph(component)
        density = nx.density(sub)  # edges / possible edges = |E| / (|V|(|V|-1)/2)
      
        if density < 0.30:  # must be meaningfully dense
            continue
      
        members = [account_ids[n] for n in component]
        degrees = dict(sub.degree())
        hub_idx = max(degrees, key=degrees.get)   # highest-degree node = coordinator
        hub_id = account_ids[hub_idx]
      
        roles = {m: 'member' for m in members}
        roles[hub_id] = 'coordinator'
      
        rings.append(Ring(
            ring_id = f'CLUSTER_{hub_id}',
            shape = 'CLUSTER',
            confidence = min(0.30 + density * 0.60, 0.95),
            ...
        ))
    return rings
```

**Why density threshold = 0.30 (not 0.40 or 0.50)?**

This was a deliberate fix (noted in the code as "FIX: density threshold lowered from 0.4 → 0.3"). The KNN graph uses cosine similarity to connect similar accounts. In a cosine-similarity-based graph, edges represent behavioral similarity, not actual transactions. Behaviorally similar legitimate accounts (e.g., many students from the same university area) might form components with density 0.25–0.35. The original threshold of 0.40 was too strict and missed some real clusters.

After analysis of false positive rates at various thresholds:

* 0.20: too many false positives (catches legitimate peer groups)
* 0.30: good balance — catches real clusters while avoiding peer groups
* 0.40: too strict — misses looser but still criminal clusters

**Graph density formula:**

```
density = |E| / (|V| × (|V|-1) / 2)   [for undirected graph]
density ∈ [0, 1]   where 0 = no edges, 1 = all possible edges present
```

---

### 18.3 Ring Detector: Detection Order and Time Budget

```python
def detect_all_rings(G, account_ids, gnn_scores=None):
    start_total = time.time()
    all_rings = []
    seen_accounts = set()  # track which account sets already form a ring
  
    def is_duplicate(ring):
        key = frozenset(ring.accounts)
        if key in seen_accounts:
            return True
        seen_accounts.add(key)
        return False
  
    # CYCLE first: most definitive, SCC algorithm is O(V+E) total
    for r in _detect_cycles(G, account_ids):
        if not is_duplicate(r):
            all_rings.append(r)
  
    # STAR: only high-risk nodes as candidates
    for node in G.nodes():
        if time.time() - start_total > 60:  # 60-second budget for all STAR checks
            break
        risk = G.nodes[node].get('risk', 0.0)
        if risk < 0.50:                     # skip low-risk hub candidates
            continue
        ring = _detect_star(G, node, account_ids, min_spokes=5)
        if ring and not is_duplicate(ring):
            all_rings.append(ring)
  
    # CHAIN: start from source nodes (in_degree=0)
    for node in G.nodes():
        if time.time() - start_total > 90:
            break
        if G.in_degree(node) == 0:
            ring = _detect_chain(G, node, account_ids)
            if ring and not is_duplicate(ring):
                all_rings.append(ring)
  
    # CLUSTER: density-based
    for r in _detect_clusters(G, account_ids):
        if not is_duplicate(r):
            all_rings.append(r)
  
    # Annotate fraud rates from GNN scores
    if gnn_scores is not None:
        id_to_idx = {aid: i for i, aid in enumerate(account_ids)}
        for ring in all_rings:
            scores = [gnn_scores[id_to_idx[a]] for a in ring.accounts if a in id_to_idx]
            ring.fraud_rate = float(np.mean([s > 0.5 for s in scores])) if scores else 0.0
  
    return all_rings
```

**Why this detection order?**

1. **CYCLE first:** SCC algorithm runs in O(V+E) once for the entire graph — fastest and most definitive. Cycles are the clearest fraud structure.
2. **STAR second:** Requires checking individual nodes; filtered by GNN risk to reduce search space. Handles the most common real-world mule topology.
3. **CHAIN third:** DFS from source nodes; naturally bounded by MAX_HOPS=6.
4. **CLUSTER last:** Most permissive criterion; runs on the complete graph structure.

**Deduplication:** `frozenset(ring.accounts)` ensures the same set of accounts isn't counted as multiple rings (e.g., a CYCLE that's also inside a CLUSTER).

### 18.4 Saving and Loading Ring Cache

```python
def save_rings_to_artifacts(rings):
    ring_dicts = [
        {
            'ring_id': r.ring_id,
            'shape': r.shape,
            'accounts': r.accounts,
            'roles': r.roles,
            'hub_node': r.hub_node,
            'fraud_rate': r.fraud_rate,
            'confidence': r.confidence,
            'size': len(r.accounts)
        }
        for r in rings
    ]
    with open(ARTIFACTS_DIR / 'rings_cache.json', 'w') as f:
        json.dump({'rings': ring_dicts, 'total': len(ring_dicts)}, f, indent=2)
```

At inference time in `api/routes/score.py`:

```python
def _get_ring_membership(account_id):
    rings = _load_rings_cache()  # loads from rings_cache.json (cached in memory)
    for ring in rings:
        if account_id in ring.get('accounts', []):
            return True
    return False
```

**The demo fallback (`api/routes/rings.py`):** If `rings_cache.json` doesn't exist (training hasn't run), the API serves hardcoded demo rings (STAR, CYCLE, CHAIN, CLUSTER typologies) with realistic fraud rates and confidence scores. The source field in the response is "demo" vs "trained" to distinguish.

---

## SECTION 19 — LOUVAIN COMMUNITY DETECTION

### 19.1 What Louvain Does

Louvain community detection partitions the account graph into communities — groups of accounts that are more densely connected internally than externally. In our context, communities represent clusters of accounts that transaction with each other, share devices, or are behaviorally similar (via KNN edges).

**File:** `ai-engine/models/community/louvain.py`

### 19.2 The Modularity Objective

Louvain maximizes  **network modularity Q** :

```
Q = (1/2m) × Σᵢⱼ [Aᵢⱼ - kᵢkⱼ/2m] × δ(cᵢ, cⱼ)
```

Where:

* `m` = total number of edges in the graph
* `Aᵢⱼ` = 1 if there is an edge between nodes i and j, 0 otherwise
* `kᵢ` = degree of node i (number of edges)
* `kᵢkⱼ/2m` = expected number of edges between i and j in a random graph with same degree sequence
* `δ(cᵢ, cⱼ)` = 1 if i and j are in the same community, 0 otherwise

**Intuition:** Modularity Q measures how much more densely connected a community is internally than you'd expect by random chance, summed over all communities. High Q means each community has many more internal edges than expected randomly — strong community structure.

**Q ranges:**

* Q > 0.3: significant community structure
* Q > 0.5: strong community structure
* Q = 0: no better than random

### 19.3 The Louvain Algorithm

Louvain runs in two iterating phases:

**Phase 1 — Greedy modularity optimization:**

1. Assign each node to its own community (N communities for N nodes)
2. For each node, try moving it to each neighboring node's community
3. Move to the community that gives maximum ΔQ (modularity gain)
4. Repeat until no move improves Q

```
ΔQ = [Σᵢₙ + 2kᵢ,ᵢₙ]/2m - [(Σₜₒₜ + kᵢ)/2m]² - [Σᵢₙ/2m - (Σₜₒₜ/2m)² - (kᵢ/2m)²]
```

Where `Σᵢₙ` = internal edge weights, `Σₜₒₜ` = total edge weights incident to community, `kᵢ,ᵢₙ` = edges from i to community.

**Phase 2 — Community aggregation:**

* Collapse each community into a single "super-node"
* Edge weight between super-nodes = sum of edges between their constituent nodes
* Return to Phase 1 on the reduced graph

This hierarchical approach allows Louvain to handle both micro-scale community structure (small tight rings) and macro-scale structure (large loose fraud networks).

### 19.4 Community Fraud Rate Assignment

```python
def detect_communities(G, account_ids, gnn_scores=None, labels=None):
    undirected = G.to_undirected()
  
    # Run Louvain
    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(undirected, resolution=1.0, random_state=42)
    else:
        # Fallback: greedy modularity communities
        communities = nx.algorithms.community.greedy_modularity_communities(undirected)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
  
    # Map node index → account_id
    account_community = {account_ids[node]: cid for node, cid in partition.items()}
  
    # Compute fraud rate per community
    community_members = {}
    for aid, cid in account_community.items():
        community_members.setdefault(cid, []).append(aid)
  
    id_to_idx = {aid: i for i, aid in enumerate(account_ids)}
  
    community_fraud_rate = {}
    for cid, members in community_members.items():
        if gnn_scores is not None:
            scores = [float(gnn_scores[id_to_idx[a]]) for a in members if a in id_to_idx]
            rate = float(np.mean([s > 0.5 for s in scores])) if scores else 0.0
        elif labels is not None:
            labs = [int(labels[id_to_idx[a]]) for a in members if a in id_to_idx]
            rate = float(np.mean(labs)) if labs else 0.0
        else:
            rate = 0.0
        community_fraud_rate[cid] = rate
  
    # Each account gets their community's fraud rate
    account_fraud_rate = {aid: community_fraud_rate[cid] for aid, cid in account_community.items()}
  
    return {
        'account_community': account_community,
        'community_fraud_rate': {str(k): v for k, v in community_fraud_rate.items()},
        'account_fraud_rate': account_fraud_rate,
        'n_communities': len(community_members)
    }
```

**Why use GNN scores to compute community fraud rate (not training labels)?**

Training labels only have 81 confirmed fraud accounts. Many community members will be unlabeled. If we compute community fraud rate only from labeled accounts, a community with 200 members but only 3 labeled fraud accounts would get rate = 1.5% — misleadingly low.

GNN scores cover all 9,082 accounts. Even unlabeled accounts get a fraud probability from the GNN. Using `gnn_score > 0.50` as the fraud criterion for community rate calculation captures the full picture of suspected fraud across the community, not just confirmed cases.

**Resolution parameter = 1.0:** Louvain's resolution parameter controls community granularity. Higher values favor smaller, more granular communities; lower values favor fewer, larger communities. 1.0 is the standard, balanced setting.

### 19.5 Community Data Access at Inference

```python
# From api/routes/score.py
def _get_community_fraud_rate(account_id):
    cache = _load_community_cache()   # loads community_data.json
    return float(cache.get(account_id, 0.0))
```

If an account is new (not in the graph during training), it gets `community_fraud_rate = 0.0` — no community context available. This is handled gracefully.

---

## SECTION 20 — PAGERANK HUB SCORING

### 20.1 The Algorithm

PageRank (Page et al., 1999 — the original Google algorithm) measures the "importance" of a node in a directed graph based on the importance of nodes that link to it.

**The formula:**

```
PR(v) = (1-α)/N + α × Σ_{u ∈ N⁻(v)} PR(u)/|N⁺(u)|
```

Where:

* `α = 0.85` = damping factor (teleportation probability = 1-α = 0.15)
* `N` = total number of nodes
* `N⁻(v)` = nodes that have edges pointing TO node v (in-neighbors)
* `N⁺(u)` = number of edges going OUT from node u (out-degree)

**Intuition:** Imagine a random walk on the graph. At each step, with probability α=0.85 you follow an outgoing edge; with probability 1-α=0.15 you "teleport" to a random node. PageRank = the fraction of time a random walker spends at each node in the long run.

**In money laundering context:** High PageRank means many accounts send money through you. An orchestrator receiving from many sources has high in-degree from high-PageRank nodes (those sources also receive from many places), giving it very high PageRank. This directly maps to the "hub" role in a mule network.

### 20.2 Normalization

```python
def compute_pagerank(G, account_ids, alpha=0.85, max_iter=100):
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=1e-6)
  
    # Map node index → account_id
    raw = {account_ids[node]: score for node, score in pr.items() if node < len(account_ids)}
  
    if not raw:
        return {}
  
    # Normalize to [0, 1]
    max_pr = max(raw.values())
    min_pr = min(raw.values())
    span = max_pr - min_pr or 1.0
  
    normalized = {aid: (score - min_pr) / span for aid, score in raw.items()}
    return normalized
```

**Why normalize?** Raw PageRank values are very small (for N=9,082 nodes, average raw PR ≈ 1/9082 ≈ 0.00011). Normalizing to [0,1] makes the values interpretable and comparable across different graph sizes.

### 20.3 Role Classification

```python
def classify_node_role(account_id, pagerank_score, out_degree, in_degree, gnn_score=0.0):
    if pagerank_score > 0.8 and out_degree >= 5:
        return 'orchestrator'
    if pagerank_score > 0.6 and out_degree >= 3:
        return 'hub'
    if pagerank_score > 0.4 and in_degree >= 2 and out_degree >= 2:
        return 'bridge'
    if in_degree >= 2 and out_degree == 0:
        return 'terminal'  # cash-out point: receives but never sends
    if gnn_score > 0.6:
        return 'mule'
    return 'legitimate'
```

**Role definitions:**

| Role         | PageRank | Out-degree | In-degree | Meaning                                   |
| ------------ | -------- | ---------- | --------- | ----------------------------------------- |
| orchestrator | > 0.8    | ≥ 5       | any       | Top-level criminal controller             |
| hub          | > 0.6    | ≥ 3       | any       | Secondary distributor                     |
| bridge       | > 0.4    | ≥ 2       | ≥ 2      | Connects subnetworks                      |
| terminal     | any      | 0          | ≥ 2      | ATM cash-out or crypto offramp            |
| mule         | any      | any        | any       | GNN-flagged but not topologically special |
| legitimate   | low      | low        | low       | No suspicious signals                     |

**Why out_degree=0 for terminal?**
A terminal account receives money and never sends it further (out_degree=0). In practice this is the cash-out point — money is withdrawn as cash or converted to cryptocurrency. The account's value to the criminal network is purely as a destination, not a relay.

---

## SECTION 21 — CYBERSECURITY ENGINE (BEI): ARCHITECTURE OVERVIEW

### 21.1 What the BEI Engine Does

The Browser Entropy Index (BEI) engine (`cybersec-engine/`, Port 8002) captures device-level and behavioral signals that are completely invisible to transaction-based analysis:

* A bot or RAT operating 15 accounts simultaneously produces statistically identical mouse movement patterns across all sessions
* The same physical device accessing 12 accounts in 2 hours is a velocity violation
* Puppeteer headless Chrome has a characteristic WebGL renderer string
* Dridex malware has a known JA3 TLS fingerprint hash

These signals are orthogonal to transaction features — they detect the *mechanism of control* rather than the  *pattern of transactions* .

### 21.2 BEI API Endpoint

```python
# From cybersec-engine/api/routes.py

class DeviceSignal(BaseModel):
    account_id: Optional[str] = 'unknown'
    canvas_hash: Optional[str] = ''
    webgl: Optional[str] = ''
    screen: Optional[dict] = {}
    timezone: Optional[str] = ''
    language: Optional[str] = ''
    audio_hash: Optional[str] = ''
    ip: Optional[str] = '0.0.0.0'
    user_agent: Optional[str] = ''
    ja3_hash: Optional[str] = None
    keystroke_intervals_ms: Optional[list[float]] = []
    mouse_coordinates: Optional[list[list]] = []
    event_count: Optional[int] = 0
    session_duration_seconds: Optional[float] = 0.0
    form_fill_seconds: Optional[float] = None

@router.post('/v1/bei')
async def score_device(payload: DeviceSignal):
    # 1. Compute fingerprint
    fp32 = compute_fingerprint(signals)
  
    # 2. Known fraud device check (O(1) Redis lookup)
    if is_known_fraud_device(fp32):
        return {'bei_risk_score': 1.0, 'flags': ['KNOWN_FRAUD_DEVICE']}
  
    # 3. Headless browser / timezone checks
    # 4. JA3 TLS analysis
    # 5. Velocity tracking
    # 6. Device graph (controller risk)
    # 7. Behavioral biometrics
  
    # 8. Aggregate BEI score
    final_bei = max(vel_score, controller_score, bio_score, ja3_score)
  
    return {
        'bei_risk_score': round(final_bei, 4),
        'flags': flags,
        'fingerprint': fp32,
        ...
    }
```

---

## SECTION 22 — DEVICE FINGERPRINTING

### 22.1 Canvas Fingerprinting

```python
# Frontend JavaScript (fingerprint.js)
const canvas = document.createElement('canvas')
const ctx = canvas.getContext('2d')
ctx.textBaseline = 'top'
ctx.font = '14px Arial'
ctx.fillStyle = '#7B2FBE'
ctx.fillText('Nyxara🔮BEI', 2, 2)  # render specific text
ctx.fillStyle = 'rgba(192,132,252,0.5)'
ctx.fillRect(50, 1, 62, 20)         # render shapes
signals.canvas_hash = canvas.toDataURL().slice(-32)  # last 32 chars of PNG data URL
```

**Why this works:** The GPU, graphics driver, operating system, and anti-aliasing algorithm all affect exactly how pixels are rendered. The same canvas drawing instruction produces subtly different pixel values on different hardware. Even two identical-model laptops with different driver versions produce different canvas outputs. The last 32 characters of the PNG data URL capture enough entropy to distinguish devices while being compact.

**Collision probability:** Two different devices producing the same canvas hash (collision) is extremely rare — canvas fingerprinting has >99% device-uniqueness in practice.

### 22.2 WebGL Fingerprinting

```python
const gl = document.createElement('canvas').getContext('webgl')
const ext = gl?.getExtension('WEBGL_debug_renderer_info')
if (ext) {
    signals.webgl = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
    // Example: "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    // Headless Chrome: "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))"
}
```

**Headless browser detection:**

```python
def is_headless_browser(signals):
    screen = signals.get('screen', {})
    w, h = screen.get('width', 1920), screen.get('height', 1080)
    # Headless Chrome default viewport
    if w == 800 and h == 600:
        return True
    # SwiftShader = software renderer used by Puppeteer headless
    if signals.get('webgl', '') in ('', 'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero))):'):
        return True
    return False
```

**Why headless detection matters:** Headless browsers (Puppeteer, Playwright, Selenium) are used to automate form filling and account operations. A mule controller running 15 account sessions simultaneously would use headless automation rather than manually interacting with 15 browser windows.

### 22.3 Audio Fingerprinting

```python
const AudioCtx = window.AudioContext || window.webkitAudioContext
const ac = new AudioCtx()
const analyser = ac.createAnalyser()
const oscillator = ac.createOscillator()
oscillator.connect(analyser)
analyser.connect(ac.destination)
oscillator.start(0)
const data = new Float32Array(analyser.frequencyBinCount)
analyser.getFloatFrequencyData(data)
signals.audio_hash = data.slice(0, 32).reduce((a, b) => a + b, 0).toFixed(6)
```

**Why audio fingerprinting:** The Web Audio API processes signals using the device's audio hardware and software. Tiny differences in floating-point arithmetic, digital signal processing pipelines, and audio codecs produce device-unique frequency response characteristics. This adds another independent dimension of entropy to the fingerprint.

### 22.4 The FP32 Device Fingerprint

```python
# From cybersec-engine/bei/fingerprint.py
import hashlib
import json

def compute_fingerprint(signals: dict) -> str:
    # Sort keys for canonical representation
    canonical = json.dumps(signals, sort_keys=True, separators=(',', ':'))
    # SHA-256 hash, take first 32 characters
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]
```

**Why 32 characters?** SHA-256 produces 64 hex characters (256 bits). The first 32 characters (128 bits) provide 2^128 ≈ 3.4 × 10^38 possible fingerprints — far more than sufficient to uniquely identify every device ever used. Truncating to 32 chars keeps keys compact for Redis storage.

**Why SHA-256 over MD5?** MD5 is cryptographically broken (collision attacks exist). While we're not doing cryptographic authentication here, SHA-256 provides better distribution and collision resistance. The risk with MD5 is that a sophisticated attacker could engineer two different device configurations that produce the same fingerprint.

### 22.5 Timezone Mismatch Detection

```python
def timezone_mismatch(signals, registered_state=None):
    tz = signals.get('timezone', '')
    indian_timezones = {'Asia/Kolkata', 'Asia/Calcutta'}  # IST equivalents
    return bool(tz and tz not in indian_timezones)
```

**Fraud signal:** An Indian bank account being operated from a timezone outside IST is suspicious. This catches:

* VPN users operating from foreign servers (though VPN doesn't change JS timezone)
* Accounts being operated from foreign-based criminal networks
* Automated scripts that don't properly mock the timezone

**When it triggers BEI flag `TIMEZONE_MISMATCH_NON_INDIAN`:** Adds at least 0.30 to BEI score.

---

## SECTION 23 — BEHAVIORAL BIOMETRICS: COMPLETE TECHNICAL REFERENCE

**File:** `cybersec-engine/bei/biometrics.py`

### 23.1 Mouse Entropy Analysis

**Collection (JavaScript):**

```javascript
const mouseHistory = []
document.addEventListener('mousemove', (e) => {
    mouseHistory.push({ x: e.clientX, y: e.clientY })
    if (mouseHistory.length > 200) mouseHistory.shift()  // rolling window
})
signals.mouse_coordinates = mouseHistory.map(pt => [Math.round(pt.x), Math.round(pt.y)])
```

**Analysis (Python):**

```python
def analyze_mouse_entropy(mouse_coordinates: list[tuple[float, float]]) -> dict:
    if len(mouse_coordinates) < 10:
        return {'entropy_score': 0.5, 'is_bot_like': False}
  
    # Compute direction changes between consecutive points
    directions = []
    for i in range(1, len(mouse_coordinates)):
        dx = mouse_coordinates[i][0] - mouse_coordinates[i-1][0]
        dy = mouse_coordinates[i][1] - mouse_coordinates[i-1][1]
        if dx == 0 and dy == 0:
            continue
        angle = math.atan2(dy, dx)  # angle in [-π, π]
        directions.append(angle)
  
    if len(directions) < 3:
        return {'entropy_score': 0.5, 'is_bot_like': False}
  
    # Quantize angles into 16 buckets covering full 360 degrees
    buckets = [0] * 16
    for angle in directions:
        bucket = int((angle + math.pi) / (2 * math.pi) * 16) % 16
        buckets[bucket] += 1
  
    # Shannon entropy
    total = sum(buckets)
    entropy = 0.0
    for count in buckets:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
  
    # Maximum entropy for 16 equal buckets = log2(16) = 4.0 bits
    normalized = entropy / 4.0
    is_bot_like = normalized < 0.30
  
    return {
        'entropy_score': round(normalized, 3),
        'raw_entropy': round(entropy, 3),
        'is_bot_like': is_bot_like,
        'n_movements': len(directions)
    }
```

**The entropy mathematics:**

Shannon entropy: `H = -Σᵢ pᵢ log₂(pᵢ)` where pᵢ = probability of direction i

* **Maximum entropy** (all 16 directions equally likely): H = log₂(16) = 4.0 bits
  * This is a human moving randomly — high entropy
* **Minimum entropy** (only one direction): H = 0.0 bits
  * This is a straight-line bot — zero entropy

**Why 16 direction buckets?**
16 buckets ≈ 22.5° per bucket. This is fine-grained enough to detect straight-line movement (all moves go into 1-2 adjacent buckets) but coarse enough that normal minor mouse direction changes fill all buckets reasonably.

**Why entropy threshold = 0.30 (normalized)?**
Empirical tuning based on expected mouse behavior:

* Pure straight-line bot: normalized entropy ≈ 0.05–0.15 (only 1–3 buckets used)
* Script with artificial jitter: normalized entropy ≈ 0.15–0.30 (maybe 4–6 buckets)
* Human in a small window: normalized entropy ≈ 0.30–0.60 (constrained but varied)
* Human across a large screen: normalized entropy ≈ 0.60–0.90 (all directions used)

Below 0.30 = bot-like behavior.

### 23.2 Keystroke Dynamics (Coefficient of Variation)

**Collection (JavaScript):**

```javascript
const keyTimings = []
document.addEventListener('keydown', () => {
    keyTimings.push(Date.now())
    if (keyTimings.length > 100) keyTimings.shift()
})
// Convert to intervals
const intervals = []
for (let i = 1; i < keyTimings.length; i++) {
    intervals.push(keyTimings[i] - keyTimings[i-1])  // milliseconds between keystrokes
}
signals.keystroke_intervals_ms = intervals
```

**Analysis (Python):**

```python
def analyze_typing_rhythm(keystroke_intervals_ms: list[float]) -> dict:
    if len(keystroke_intervals_ms) < 5:
        return {'consistency_score': 0.5, 'mean_interval': 0.0, 'cv': 0.0, ...}
  
    intervals = [max(i, 1) for i in keystroke_intervals_ms]  # prevent division by zero
    mean_ms = statistics.mean(intervals)
    std_ms = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
  
    # Coefficient of Variation: std/mean
    # Unitless measure of relative variability
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
  
    # Map CV to consistency score (0=human, 1=bot)
    if cv < 0.05:
        consistency_score = 0.95   # near-certain automation: intervals too regular
    elif cv < 0.15:
        consistency_score = 0.75   # suspiciously consistent
    elif cv < 0.30:
        consistency_score = 0.50   # borderline
    else:
        consistency_score = 0.20   # normal human variation
  
    return {
        'consistency_score': round(consistency_score, 3),
        'mean_interval_ms': round(mean_ms, 1),
        'std_interval_ms': round(std_ms, 1),
        'cv': round(cv, 3),
        'sample_size': len(intervals)
    }
```

**The Coefficient of Variation formula:**

```
CV = σ / μ

where:
σ = standard deviation of inter-keystroke intervals
μ = mean inter-keystroke interval
```

CV is dimensionless — it normalizes variability by the mean, making it comparable across users with different average typing speeds.

**Typical CV values by user type:**

| User Type                                | CV         | Consistency Score |
| ---------------------------------------- | ---------- | ----------------- |
| Automated script (fixed 150ms intervals) | 0.00–0.02 | 0.95              |
| Trained fast typist (very regular)       | 0.05–0.12 | 0.75              |
| Average human typist                     | 0.30–0.60 | 0.20              |
| Hunt-and-peck typist                     | 0.60–1.20 | 0.20              |

**The RAT operator insight (from BioCatch research):**
A Remote Access Trojan operator controlling 15 accounts will have statistically identical CV across all sessions — they're the same human typing on the same keyboard in the same physical conditions. P(same human typing 15 accounts | CV=0.02 across all) ≈ 0.999. This makes CV a powerful signal for detecting single-operator mule farms.

### 23.3 Session Velocity Analysis

```python
def compute_session_velocity(event_count, session_duration_seconds, form_fill_seconds=None):
    if session_duration_seconds <= 0:
        return {'velocity_score': 0.0, 'events_per_second': 0.0, 'bot_fill_detected': False}
  
    eps = event_count / session_duration_seconds  # events per second
  
    # Map EPS to velocity score
    if eps > 50:    vel_score = 1.0   # definitely automated
    elif eps > 10:  vel_score = 0.7   # suspicious
    elif eps > 5:   vel_score = 0.4   # borderline
    else:           vel_score = 0.1   # normal
  
    # Form fill speed: humans take 30-120 seconds
    bot_fill = False
    if form_fill_seconds is not None and form_fill_seconds < 3.0:
        bot_fill = True
        vel_score = max(vel_score, 0.80)
  
    return {
        'velocity_score': round(vel_score, 3),
        'events_per_second': round(eps, 2),
        'session_duration_s': round(session_duration_seconds, 1),
        'bot_fill_detected': bot_fill
    }
```

**Event count includes:** keystrokes, mouse moves (sampled), clicks, scroll events. A human browsing naturally generates 1–5 events/second. A bot executing a script generates 50–200 events/second (each form field fill, each button click is near-instantaneous).

**Form fill time:** The banking form in the Analyzer page has ~8 fields. A human reads and fills each field: name (read + type), account number (read + type + verify), etc. Realistically 45–90 seconds total. A bot fills all fields simultaneously in < 1 second. The `form_fill_seconds` field captures this timing.

### 23.4 Biometric Score Aggregation

```python
def aggregate_biometric_score(typing: dict, mouse: dict, velocity: dict) -> float:
    t_score = typing.get('consistency_score', 0.5)    # 0=human, 1=bot
    m_score = 1.0 - mouse.get('entropy_score', 0.5)  # invert: low entropy = high risk
    v_score = velocity.get('velocity_score', 0.0)
  
    # Special case: simultaneous bot movement AND bot form fill = very high confidence
    if mouse.get('is_bot_like') and velocity.get('bot_fill_detected'):
        return 0.95
  
    # Weighted aggregation
    return round(0.40 * t_score + 0.30 * m_score + 0.30 * v_score, 3)
```

**Weight rationale:**

* Keystroke (40%): Most informative individual signal — timing is the hardest to fake
* Mouse entropy (30%): Strong signal but harder to collect in many scenarios (mobile devices, touchscreens)
* Velocity (30%): Easy to detect but can be gamed by adding artificial delays

**The special case (bot movement + bot form fill):** When two independent signals simultaneously confirm bot behavior, the confidence in automation is extremely high. The 0.95 score reflects this — there's essentially no legitimate explanation for straight-line mouse movement combined with < 3 second form fill.

---

## SECTION 24 — JA3 TLS FINGERPRINTING

### 24.1 What JA3 Is

**File:** `cybersec-engine/ja3/fingerprint.py`

JA3 is a method developed by John Althouse at Salesforce (2017) to fingerprint TLS client implementations by hashing the parameters in the TLS ClientHello message.

When a client (browser, app, malware) initiates a TLS connection, it sends a ClientHello message containing:

* TLS version (e.g., 1.2, 1.3)
* Supported cipher suites (ordered list of encryption algorithms)
* TLS extensions (e.g., SNI, session tickets)
* Supported elliptic curves
* Elliptic curve point formats

Different software has characteristic ClientHello configurations. Chrome 120, Firefox 121, Safari 17, Dridex banking trojan — each produces a different combination of these parameters.

### 24.2 Computing the JA3 Hash

```python
def compute_ja3_from_params(tls_version, ciphers, extensions, elliptic_curves, elliptic_curve_points):
    raw = ','.join([
        str(tls_version),
        '-'.join(str(c) for c in ciphers),
        '-'.join(str(e) for e in extensions),
        '-'.join(str(ec) for ec in elliptic_curves),
        '-'.join(str(ep) for ep in elliptic_curve_points)
    ])
    return hashlib.md5(raw.encode()).hexdigest()  # 32-character MD5 hash
```

**Why MD5 for JA3?** This is the original JA3 specification by Althouse — MD5 was chosen for its speed and to produce a compact 32-character hash. MD5's known collision vulnerabilities don't matter here because we're using JA3 for identification, not cryptographic security. An attacker can't easily engineer a JA3 hash collision that exactly matches a legitimate browser's hash while also maintaining malware functionality.

### 24.3 Known Malware JA3 Hashes

```python
KNOWN_BAD_JA3 = {
    'e7d705a3286e19ea42f587b344ee6865',  # Trickbot (banking trojan targeting India)
    '72a589da586844d7f0818ce684948eea',  # Dridex (RAT used in mule recruitment)
    'a0e9f5d64349fb13191bc781f81f42e1',  # Cobalt Strike (pen tool used by criminal orgs)
}

KNOWN_GOOD_JA3 = {
    'cd08e31494f9531f560d64c695473da9',  # Chrome 120
    'b32309a26951912be7dba376398abc3b',  # Firefox 121
    'a35f40e4b8c8f0b0c6f2f2e6e9c1a2b3',  # Safari 17
}
```

```python
def analyze_ja3(ja3_hash: str | None) -> dict:
    if not ja3_hash:
        return {'risk': 'unknown', 'score': 0.20, 'reason': 'No JA3 hash provided'}
  
    ja3_hash = ja3_hash.lower().strip()
  
    if ja3_hash in KNOWN_BAD_JA3:
        return {'risk': 'critical', 'score': 1.0, 'reason': f'JA3 matches known malware', 'action': 'BLOCK'}
  
    if ja3_hash in KNOWN_GOOD_JA3:
        return {'risk': 'low', 'score': 0.05, 'reason': 'JA3 matches known legitimate browser', 'action': 'ALLOW'}
  
    # Unknown: conservative moderate score
    return {'risk': 'unknown', 'score': 0.15, 'reason': 'JA3 not in known-good or known-bad lists', 'action': 'MONITOR'}
```

**The `MALWARE_JA3_SIGNATURE` flag:** When a known-bad JA3 hash is detected, the flag `MALWARE_JA3_SIGNATURE` is added to `cyberFlags` and BEI is set to 1.0 — regardless of all other signals. Trickbot and Dridex are specific banking trojans that have caused significant fraud losses in India.

### 24.4 Production JA3 Collection

```python
# IMPORTANT: JavaScript in the browser CANNOT access TLS parameters
# TLS is at the network layer, below the application layer
# 
# In production: JA3 is computed by the load balancer or API gateway
# (nginx with JA3 module, Cloudflare, AWS WAF) at the TLS termination point
# and added to request headers before forwarding to the backend
# 
# Example nginx header: X-JA3-Hash: e7d705a3286e19ea42f587b344ee6865
# Backend reads: ja3_hash = request.headers.get('X-JA3-Hash')
# 
# For hackathon demo: ja3_hash is passed from the frontend device signal
# (pre-computed or set to null)
```

---

## SECTION 25 — VELOCITY TRACKING (REDIS-BACKED)

### 25.1 Architecture and Why Redis

**File:** `cybersec-engine/bei/velocity.py`

Redis is chosen specifically because:

1. **O(1) operations:** SADD (add to set) and SCARD (count set members) are both O(1) — constant time regardless of set size
2. **Automatic TTL expiry:** EXPIRE sets a time-to-live; Redis automatically cleans up expired keys — no cleanup code needed
3. **Atomic operations:** Redis commands are atomic — no race conditions when multiple BEI scoring requests happen simultaneously
4. **Throughput:** Redis handles 50,000+ operations/second on a standard server — far exceeding any bank's real-time request volume

```python
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    REDIS_AVAILABLE = True
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f'Redis unavailable. Velocity tracking disabled.')
    # Graceful degradation: velocity tracking disabled, other BEI signals still work
```

### 25.2 Device Velocity Tracking

```python
def track_device_access(fp32: str, account_id: str) -> dict:
    if not REDIS_AVAILABLE:
        return {'device_account_count': 0, 'flags': []}
  
    key = f'device:{fp32}:accounts'   # e.g., 'device:a3f4b2c1d5e6f789:accounts'
  
    pipe = r.pipeline()    # batch commands in one network round-trip
    pipe.sadd(key, account_id)   # add account to the SET for this device
    pipe.expire(key, 86400)      # 24-hour TTL (86,400 seconds)
    pipe.scard(key)              # count unique accounts
    results = pipe.execute()
    count = results[2]           # third command result = count
  
    flags = []
    if count > 10:   # HARD threshold
        flags.append(f'DEVICE_CONTROLS_{count}_ACCOUNTS_HARD_BLOCK')
    elif count > 5:  # SOFT threshold
        flags.append(f'DEVICE_CONTROLS_{count}_ACCOUNTS_SOFT_FLAG')
  
    return {'device_account_count': count, 'flags': flags}
```

**The data structure:** Redis SET (not list or sorted set). A SET automatically deduplicates — if the same account_id is added multiple times (account accessed 10 times in one day from the same device), SADD just keeps one entry. SCARD then returns the count of UNIQUE accounts, not total accesses.

**The 24-hour TTL window:** We care about how many different accounts a device accesses within a rolling 24-hour window. After 24 hours, the counter resets automatically. This prevents historical data from permanently blocking devices.

**Pipeline vs individual commands:** Without pipelining, 3 Redis commands = 3 network round-trips = ~3ms latency. With pipeline, 3 commands = 1 network round-trip = ~1ms. For high-volume scoring, this matters.

### 25.3 IP Address Velocity Tracking

```python
def track_ip_access(ip: str, account_id: str) -> dict:
    if not REDIS_AVAILABLE:
        return {'ip_account_count': 0, 'flags': []}
  
    key = f'ip:{ip}:accounts:1h'   # 1-hour window
  
    pipe = r.pipeline()
    pipe.sadd(key, account_id)
    pipe.expire(key, 3600)         # 1-hour TTL (3,600 seconds)
    pipe.scard(key)
    results = pipe.execute()
    count = results[2]
  
    flags = []
    if count > 8:   # HARD
        flags.append(f'IP_ACCESSING_{count}_ACCOUNTS_BLOCK')
    elif count > 3:  # SOFT
        flags.append(f'IP_ACCESSING_{count}_ACCOUNTS_ALERT')
  
    return {'ip_account_count': count, 'flags': flags}
```

**Why 1-hour window for IP vs 24-hour for device?**
IP-based signals change faster. A legitimate IP (e.g., an office NAT) might have hundreds of people behind it, but they won't all be accessing different bank accounts within one hour. A criminal organization might rotate devices daily but tends to access many accounts from one IP address within a short time window during an active fraud session.

The narrower 1-hour window catches active fraud sessions while reducing false positives from legitimate shared IPs.

### 25.4 The Velocity Score Formula

```python
def compute_velocity_score(device_count: int, ip_count: int) -> float:
    d_score = min(device_count / 10, 1.0)   # normalize: 10+ accounts = max score
    i_score = min(ip_count / 8, 1.0)        # normalize: 8+ accounts = max score
    return max(d_score, i_score)            # take the worse of the two
```

**Why max() not average?** Either signal alone (one device controlling 12 accounts OR one IP accessing 10 accounts) is sufficient to flag a hard block. Taking the max means the highest-risk signal wins — this is a conservative approach appropriate for fraud detection where false negatives (missing fraud) are much more costly than false positives.

### 25.5 Known Fraud Device Registry

```python
def register_fraud_device(fp32: str):
    if REDIS_AVAILABLE:
        r.sadd('fraud_devices', fp32)   # permanent SET, no TTL

def is_known_fraud_device(fp32: str) -> bool:
    if not REDIS_AVAILABLE:
        return False
    return r.sismember('fraud_devices', fp32)  # O(1) membership check
```

This is the fastest possible check — O(1) Redis SET membership. When a device is confirmed as fraud (analyst confirms the alert), its fp32 is added to the `fraud_devices` SET. Every future request from that device gets BEI = 1.0 and `KNOWN_FRAUD_DEVICE` flag before any other computation.

**The `fraud_devices` SET has no TTL** — it's permanent. Once confirmed as a fraud device, it remains blocked until manually removed.

### 25.6 Device Graph (Controller Detection)

```python
# From cybersec-engine/bei/device_graph.py
# In-memory bipartite graph: device_fp ↔ account_id

_G = nx.DiGraph()

def record_access(fp32: str, account_id: str):
    fp_node = f'fp:{fp32}'
    acct_node = f'acct:{account_id}'
    _G.add_edge(fp_node, acct_node)

def get_controller_risk_score(fp32: str) -> float:
    """High degree fingerprint node = probable mule controller."""
    accounts = get_device_controlled_accounts(fp32)
    n = len(accounts)
    if n >= 10: return 1.0
    if n >= 5:  return 0.7
    if n >= 3:  return 0.4
    return 0.0
```

This builds a bipartite graph where device fingerprints and account IDs are nodes, and edges connect devices to accounts they access. High out-degree from a device fingerprint node = controller operating many accounts. The `get_controller_risk_score()` implements step-function risk levels based on controlled account count.

**In-memory vs Redis-backed:** The device graph is in-memory for the hackathon. The comment notes "Persist to MongoDB in production" — the in-memory graph is lost on restart. In production, each `record_access()` call would write to MongoDB, and the graph would be reconstructed from the database at startup.

### 25.7 Complete BEI Final Score Assembly

```python
# From cybersec-engine/api/routes.py (simplified)

# After computing all signals:
vel_score = compute_velocity_score(dev_vel['device_account_count'], ip_vel['ip_account_count'])
bio_score = aggregate_biometric_score(typing, mouse, velocity)

# Final BEI: maximum across all signal sources
bei_score = max(vel_score, controller_score, bio_score, ja3_result['score'])

# Additional adjustments for specific flags
if 'HEADLESS_BROWSER_DETECTED' in flags:
    bei_score = max(bei_score, 0.60)   # headless alone warrants FLAG-level BEI

if 'TIMEZONE_MISMATCH_NON_INDIAN' in flags:
    bei_score = max(bei_score, 0.30)   # timezone alone warrants elevated BEI

return {
    'bei_risk_score': round(bei_score, 4),
    'flags': flags,
    'fingerprint': fp32,
    'device_account_count': dev_vel['device_account_count'],
    'ip_account_count': ip_vel['ip_account_count'],
    'biometrics': {'typing': typing, 'mouse': mouse, 'velocity': velocity}
}
```

**The max() aggregation:** For BEI, the maximum across all signal sources is used (not a weighted average). Rationale: any single strong signal is sufficient to flag behavioral risk. A device might have perfectly human-like mouse movements (bio_score = 0.1) but if its JA3 hash matches Trickbot (ja3_score = 1.0), the BEI should be 1.0. Taking a weighted average would dilute this critical signal.

---

## SECTION 26 — KEY JURY QUESTIONS FOR PART 3

**Q: Why build a separate Cybersecurity Engine microservice instead of including BEI in the AI Engine?**
A: Three reasons. First, the Cybersecurity Engine depends on Redis for velocity tracking — the AI Engine depends on PyTorch and sklearn. These have very different infrastructure requirements and scaling characteristics. Second, the Cybersecurity Engine runs in real-time on every user session (it needs < 50ms), while the AI Engine does heavier batch inference. Separating them allows the BEI to scale independently based on traffic. Third, the BEI signals are fundamentally different from transaction features — they require web session context (user agent, canvas hash, keystroke timings) that is collected in real-time and is meaningless to the batch training pipeline. Separating concerns makes both services cleaner and more maintainable.

**Q: Your device graph is in-memory. What happens when the service restarts?**
A: You're right — the in-memory graph is lost on restart, which is a known limitation noted in the code comment ("Persist to MongoDB in production"). For the hackathon, the device graph resets on restart but is rebuilt within the first few hours of operation as requests come in. In production deployment, we would persist each `record_access()` call to MongoDB and reconstruct the bipartite graph from the database at startup. The Redis-backed velocity tracking IS persistent (Redis has AOF/RDB persistence modes), so the velocity counters survive restarts. The device graph is the one component that would genuinely benefit from persistent storage in a production system.

**Q: Can a sophisticated attacker evade the BEI layer?**
A: Yes, partially. A determined attacker could: (1) use a real browser with headless detection disabled (requires code modification to Chrome/Chromium), (2) inject artificial random delays into keystroke timing to increase CV above human thresholds, (3) route through residential proxy networks to spread IP-level velocity, (4) use different physical devices for each account. However, the BEI layer is not the last line of defense — it contributes only 12% to finalRisk. The GNN (35%), Ensemble (25%), and VAE (20%) layers use transaction patterns that are much harder to fake while still committing the crime. A mule account that perfectly evades BEI detection but still has near-1.0 pass-through ratio, dormancy activation pattern, and occupational velocity anomaly will still score HIGH or BLOCK. BEI is one layer in a defense-in-depth architecture, not a complete solution.

**Q: How does JA3 fingerprinting work at the TLS layer? Can the frontend collect it?**
A: JA3 operates at the TLS layer, below the HTTP/application layer. The TLS ClientHello message is sent before any HTTP content — it's part of the TLS handshake that happens when the TCP connection is first established. JavaScript running in the browser operates at the application layer (HTTP) and cannot access TLS parameters. In a production deployment, JA3 is computed by the load balancer, reverse proxy (nginx with JA3 module), or CDN (Cloudflare has native JA3) at the TLS termination point. The computed hash is then injected as an HTTP header (X-JA3-Hash) and forwarded to the backend. For the hackathon demo, we accept the ja3_hash as a parameter in the device signal payload — it would be null for most demo requests and only non-null if explicitly set for testing.

**Q: What is the difference between your velocity tracking and simple rate limiting?**
A: Standard rate limiting asks: "How many requests has this IP made in the last hour?" Velocity tracking asks: "How many DIFFERENT bank accounts has this device/IP accessed in the last hour/day?" These answer completely different questions. A legitimate power user might make 500 requests in an hour but they all relate to the same 2–3 accounts (their own and their family's). A mule controller makes 15 requests in an hour — one for each of 15 different accounts. The cardinality of accounts accessed (not the request count) is the signal. This requires a Redis SET (which deduplicates) rather than a simple counter. It's a fundamentally different metric that specifically targets the mule-controller use case.

**Q: Explain Louvain community detection and why it's better than k-means for this use case.**
A: Louvain maximizes network modularity — a graph-theoretic measure of how much more densely connected a community is internally than expected from random edge distribution. It's specifically designed for graph data where the "similarity" between nodes is edge connectivity, not Euclidean distance. K-means requires you to: (1) specify k (number of clusters) upfront — we don't know how many mule communities exist; (2) operate on Euclidean feature space — but we want communities based on actual transaction connections; (3) compute distances between feature vectors — meaningless for network structure. Louvain discovers the number of communities automatically, finds them based on actual edge topology, and produces communities that directly correspond to groups of accounts that transact with each other. A Louvain community of 300 accounts all connected by transactions is immediately interpretable as a money laundering network; a k-means cluster of 300 accounts with similar velocity values is not.

---

*Continue to Part 4 → System Architecture · 5 Microservices · 12-Step Pipeline · Blockchain Audit · Compliance/STR · Frontend Architecture · Full Performance Benchmarks · Complete Jury Q&A Bank*
