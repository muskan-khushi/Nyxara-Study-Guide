# 🔮 NYXARA — GRAND FINALE MASTERY GUIDE

## Part 1 of 4: Problem Domain · Dataset · Feature Engineering Pipeline

> **How to use this guide:** Read every word. Don't skim. Every section is something a jury member at IIT Hyderabad could ask you. The jury will include banking technologists, ML researchers, and compliance officers — so questions will span from "explain backpropagation" to "what does PMLA Section 12(1)(b) require?" This guide covers all of it at PhD depth.

---

## TABLE OF CONTENTS (All 4 Parts)

| Part                        | Contents                                                                                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Part 1**(this file) | Problem domain · Dataset anatomy · 18 key features decoded · 5-stage feature pipeline · 12 composite features                                                         |
| **Part 2**            | 6-Layer AI detection stack · GNN architecture · VAE · Contrastive learning · Ensemble · SHAP · Risk fusion formula · Adaptive threshold calibration                |
| **Part 3**            | Graph intelligence · Ring detector · Louvain · PageRank · Cybersecurity BEI layer · JA3 · Behavioral biometrics · Velocity tracking                                |
| **Part 4**            | System architecture · 5 microservices · 12-step pipeline · Blockchain audit · Compliance/STR · Frontend architecture · Performance benchmarks · Full jury Q&A bank |

---

## SECTION 1 — WHAT IS NYXARA? START HERE.

### 1.1 The One-Paragraph Definition

Nyxara is a  **production-grade, open-source AI platform that automatically classifies bank accounts as mule accounts or legitimate accounts** . A "mule account" is a bank account that criminals use to receive stolen money and forward it further, obscuring the trail. Nyxara detects these accounts using six simultaneous AI techniques — graph neural networks, contrastive transformers, gradient boosting ensembles, variational autoencoders, behavioral biometrics, and SHAP explainability — and flags them for compliance officers before significant financial damage is done.

Built for: **PSB Cybersecurity, Fraud & AI Hackathon 2026 (Bank of India × IIT Hyderabad, Problem Statement 2)**

Dataset: **9,082 bank accounts × 3,924 anonymized features**

Results: **AUC-ROC 0.982 · F1 0.910 · Precision 0.940 · Recall 0.890**

Infrastructure cost: **₹0**

### 1.2 The One-Sentence Pitch (Memorize This)

> *"Nyxara detects mule accounts by fusing graph neural network topology analysis, classical ML ensemble scoring, variational autoencoder anomaly detection, and behavioral biometric signals — all in under 100ms per account — with every decision explainable via SHAP values, narrated in FIU-IND compliant language by an LLM, and cryptographically sealed in a Merkle audit chain."*

---

## SECTION 2 — THE PROBLEM: INDIA'S MULE ACCOUNT CRISIS

### 2.1 Scale of the Problem

* **₹14.57 billion** lost to digital payment fraud in India in FY2024
* Card/internet fraud share rose from **49% (FY23) to 80% (FY24)**
* **4.5 lakh mule accounts** were frozen by government agencies in 2025
* The RBI CGM reported **95% accuracy** for MuleHunter.AI on Canara Bank data (August 2025)
* **23 banks** now use the MuleHunter system
* I4C-RBIH MoU signed **May 12, 2026** establishing MuleHunter.AI as the national standard

### 2.2 What Is a Mule Account? (Explain From First Principles)

A **money mule account** is a bank account — usually belonging to a real, often innocent person — that criminals use as an intermediate stop for stolen funds. Here's how the crime works:

```
Step 1: Criminal steals money (phishing, UPI fraud, OTP scam, etc.)
Step 2: Stolen funds sent to Mule Account A (the "hub")
Step 3: Hub immediately splits money across 5–10 Mule Accounts (the "spokes")
Step 4: Each spoke forwards funds further, often to crypto exchanges or cash ATMs
Step 5: Money is now "laundered" — origin obscured by multiple hops
```

**Why ordinary people become mules:**

* Students offered "easy commissions" to receive and forward money
* Housewives convinced it's legitimate import/export work
* Unemployed individuals desperate for income
* Sometimes accounts are taken over via SIM swap or RAT malware — the account holder doesn't even know

**Why this is hard to detect:**

* Each individual transaction may look legitimate (reasonable amount, standard timing)
* The mule account holder often has no criminal history
* The pattern is only visible when you look at the *network* — who sends to whom, and when

### 2.3 Why Existing AML (Anti-Money Laundering) Systems Fail

| Failure Mode                    | What Happens                              | Why It's Fatal                                                                  |
| ------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **Rule-based thresholds** | "Flag transactions above ₹10 lakh"       | Criminals split transactions into ₹9,900 (smurfing)                            |
| **Per-account ML**        | Each account analyzed independently       | Misses the network — a clean-looking hub account orchestrates 10 mules         |
| **Single-bank view**      | Bank A cannot see Bank B's data           | Laundering loops spanning SBI → HDFC → ICICI are invisible to any single bank |
| **No behavioral layer**   | Only transaction logs analyzed            | Bot-controlled accounts look identical to humans in transaction data            |
| **Manual STR filing**     | Compliance officers file reports manually | 7–14 day backlog destroys deterrence                                           |
| **Opaque models**         | "Black box" predictions                   | Compliance officers can't sign off on unexplained decisions                     |

### 2.4 The I4C MuleHunter.AI Benchmark

The Indian Cyber Crime Coordination Centre (I4C) and the Reserve Bank Innovation Hub (RBIH) established **MuleHunter.AI** as the national automated mule detection standard. Key benchmarks:

* AUC-ROC: ~0.95 (Canara Bank, August 2025)
* **Nyxara's AUC: 0.982 — surpasses the national benchmark**

The May 2026 MoU between I4C and RBIH specifically calls for:

1. A **Suspect Registry** for cross-bank flagged account sharing
2. **Automated detection** with standardized benchmarks
3. **Privacy-preserving intelligence sharing** protocols
4. Integration with **FIU-IND STR reporting** workflows

Nyxara's architecture addresses every single one of these requirements.

---

## SECTION 3 — THE DATASET: EVERYTHING YOU NEED TO KNOW

### 3.1 Raw Data Facts

```
File:          DataSet.xlsx
Rows:          9,082 (bank accounts)
Columns:       3,924 (anonymized features F1–F3924)
Target column: F3924 (0 = legitimate, 1 = suspicious/mule)
Fraud rate:    ~0.89% (approximately 81 confirmed mule accounts)
Column types:  ~2,949 float64, ~968 int64, ~6 string, ~1 datetime
Loading:       pandas.read_excel(path, engine='openpyxl')
```

**Critical observation:** The first ~500 rows all have label=0. This is NOT a data error — fraud accounts are concentrated in specific regions of the dataset. Always run `y.value_counts()` before any analysis.

### 3.2 Why Feature Names Are Anonymized (F1–F3924)

The bank anonymized feature names to:

1. Protect their internal feature engineering intellectual property
2. Prevent reverse-engineering of their rule-based detection system
3. Comply with data sharing agreements for the hackathon

The 18 key features were specifically identified by the bank as the most important. We **reverse-engineered** their real-world meaning by analyzing value ranges, distributions, correlations with the target, and cross-referencing with known AML indicators from FATF guidelines and industry literature.

### 3.3 The Fundamental Challenge: 0.89% Fraud Rate

This is the central problem that shapes every modeling decision:

**The Stupid Baseline:** A model that predicts "legitimate" for *every single account* achieves  **99.11% accuracy** . But it has **0% recall** — it never catches a single mule. This is why **accuracy is completely useless** as a metric here.

We use instead:

* **AUC-ROC** : Threshold-independent ranking metric. Probability that a randomly chosen fraud account scores higher than a randomly chosen legitimate account.
* **F1 Score** : Harmonic mean of Precision and Recall. Penalizes extreme values — you can't game it by having perfect precision at 0 recall.
* **Precision** : Of all accounts predicted fraud, what fraction actually are? (False alarm rate)
* **Recall** : Of all actual fraud accounts, what fraction did we catch? (Miss rate)

---

## SECTION 4 — THE 18 BANK-IDENTIFIED KEY FEATURES: COMPLETE DECODE

These 18 features are **force-included at every stage** of the 5-stage feature selection pipeline. They cannot be eliminated by any statistical criterion. Know every single one cold.

### F115 — Transaction-to-Peer Velocity Ratio

* **Observed range:** 0.3–1.5 float
* **Real meaning:** How much this account transacts compared to its occupation peer group. Value of 1.4 means 40% above peers.
* **Fraud signal:** Mule accounts transact far above their income-class peers. A student with F115=1.4 is receiving and forwarding far more money than a typical student — financial impossibility signal.
* **FATF alignment:** Corroborates ML-02 (unusual transaction volumes)

### F321 — Average Transaction Amount vs Account-Type Baseline

* **Observed range:** 0.7–1.7 float
* **Real meaning:** Ratio of this account's average transaction amount to the baseline for accounts of the same type (savings/current/etc.)
* **Fraud signal:** Extreme or perfectly round transaction amounts signal either smurfing (splitting) or pass-through behavior. Mule controllers often instruct mules to transact in round numbers below CTR thresholds.

### F527 — Debit-to-Credit Flow Ratio (Pass-Through Ratio) ⭐ MOST IMPORTANT

* **Observed range:** 0.5–1.7 float
* **Real meaning:** Ratio of total debits to total credits in the observation window
* **Fraud signal:** A value near 1.0 is the **conduit signature** — money enters and exits with near-zero net retention. A legitimate savings account accumulates money (ratio < 1.0). A mule account passes money through (ratio ≈ 1.0). Our composite feature `pass_through_score` expands on this.
* **Model impact:** Highest mean |SHAP| value in our model:  **0.284** . The single most predictive feature.
* **FATF alignment:** ML-02 (pass-through accounts)

### F531 — Network Co-Transaction Score

* **Observed range:** 0.5–1.8 float
* **Real meaning:** How many transaction counterparties this account shares with other flagged/suspicious accounts in the dataset
* **Fraud signal:** High score means this account transacts with the same entities as other suspicious accounts — direct ring membership signal. A legitimate salaried person shares counterparties with their employer, utilities, maybe a few shops. A ring member shares counterparties with the hub and other mules.

### F670 — Legacy Dormancy / Risk Flag

* **Observed range:** Binary 0/1
* **Real meaning:** Trigger from a legacy rule-based detection system — possibly a dormancy flag or a prior suspicious activity flag
* **Fraud signal:** When 1, something in the bank's existing rules triggered. Useful as corroborating evidence even if low precision on its own.

### F1692 — Linked Accounts Count

* **Observed range:** 0–10 integer
* **Real meaning:** Number of accounts sharing device fingerprints, phone numbers, or other linking signals with this account
* **Fraud signal:** A legitimate customer shares their device with maybe 1 family member. A mule controller accessing 15 accounts from one device produces F1692 >> 3 across all linked accounts. Threshold: >3 = organized network, >7 = probable mule cell.
* **Model impact:** Used directly in `network_centrality_proxy` composite feature

### F2082 — International Transaction Exposure Ratio

* **Observed range:** 0.0–small float (usually 0 for domestic customers)
* **Real meaning:** Proportion of transactions involving foreign counterparties or international transfers
* **Fraud signal:** ANY non-zero value for student/housewife/retired is a near-impossibility. These occupations have legitimate reasons for international transfers in maybe 1–2% of cases. Cross-border exposure from these groups is FATF ML-04.
* **FATF alignment:** ML-04 (international transaction exposure)
* **Composite use:** `cross_border_occupation_risk = F2082 × risk_weight[occupation]`

### F2122 — Cash Deposit Frequency / ATM Withdrawal Ratio

* **Observed range:** 0.0–0.5 float
* **Real meaning:** Proportion of transactions that are cash deposits or ATM withdrawals (as opposed to digital transfers)
* **Fraud signal:** High value = **structuring** below CTR (Currency Transaction Report) threshold. The CTR threshold in India is ₹10,000 for cash. Criminals instruct mules to make many cash deposits of ₹9,900 — this pattern is called "smurfing." High F2122 + high transaction count = classic smurfing.
* **FATF alignment:** ML-08 (structured cash deposits)

### F2582 — Occupation Peer Spending Deviation (Z-Score)

* **Observed range:** −2 to +0.5 z-score
* **Real meaning:** How many standard deviations this account's transaction amounts deviate from the median for its occupation peer group
* **Fraud signal:** A sudden large positive spike means this account is transacting far above its occupation's statistical norm — the signature of dormant mule activation. A student account that suddenly shows F2582 = +1.8 has just been activated for mule use.

### F2678 — Time-of-Day / Day-of-Week Anomaly (Z-Score)

* **Observed range:** −2 to +0.5 z-score
* **Real meaning:** Deviation in transaction timing patterns from occupation peer norms
* **Fraud signal:** Large negative value = nocturnal activity far outside peer norms. Bot-controlled or RAT-hijacked accounts transact at 3 AM IST when the declared account holder (student, housewife, retired person) would normally be asleep. In our Attack Replay scenario, nocturnal ratio = 100% is a key BLOCK signal.

### F2737 — Balance Volatility / Amount Entropy

* **Observed range:** −1.0 to +0.5
* **Real meaning:** Measure of how volatile the account balance is, or how much entropy exists in transaction amounts
* **Fraud signal:** High volatility + near-zero net balance =  **pass-through mule** . Money sloshes in and out but the account never accumulates wealth. Used in `pass_through_score = F527 × |F2737|`.

### F2956 — Internal Credit Risk / Bureau Score

* **Observed range:** 36–100 score
* **Real meaning:** Internal credit risk score or external bureau score (CIBIL equivalent)
* **Fraud signal:** Low score (<50) = financially stressed individual — the  **classic mule-recruit profile** . Criminals specifically target people who are desperate for income. A student with a low credit score is a prime target for "easy money" recruitment schemes.

### F3043 — Account Tenure / Days Since Last Activity ⭐ NULL IS THE SIGNAL

* **Observed range:** 4–114 days; **67% NULL**
* **Real meaning:** Days since the account was opened, or days since last transaction
* **Fraud signal:** **THE NULL ITSELF IS THE FRAUD SIGNAL.** 67% of accounts have missing tenure. Accounts with no recorded history are likely ghost accounts (opened specifically for mule use) or accounts that have never been used until being activated for a crime. We create `F3043_missing` binary flag (1 = tenure is NULL) which is itself a strong predictor.
* **FATF alignment:** ML-06 (ghost accounts / missing account history)

### F3836 — Account Balance / Cumulative Transaction Amount

* **Observed range:** −₹4B to +₹673M (massive range)
* **Real meaning:** Current account balance or sum of all transactions in the observation window
* **Fraud signal:** Extremely large positive for student/housewife/retired =  **financial impossibility** . Our composite `financial_impossibility_score = |F3836| / occupation_income_proxy` quantifies exactly how impossible the amount is. A student (income proxy ₹50K/year) with F3836 = ₹4.5L scores 9.0 — this is nine times their annual income in a single observation window.

### F3887 — Customer Age / Transaction Count Proxy

* **Observed range:** 51–172 integer
* **Real meaning:** Likely customer age in some transformed form, or total transaction count proxy
* **Fraud signal:** Very young customers (<25 in whatever the unit is) combined with high F3894 (transaction velocity) = recruited student mule pattern. Young people are specifically targeted because they're more likely to accept "easy money" offers and less likely to ask questions.

### F3889 — Branch-Product Code Composite (String)

* **Observed range:** Categorical strings like "G365D", "B210A", "K401C"
* **Real meaning:** A composite code identifying the bank branch where the account was opened and possibly the account product type
* **Fraud signal:** Certain branches are **fraud hotspots** where mule recruitment campaigns are active. In our FraudMap visualization: branch G365D has 81% fraud rate, B210A has 67%. These branches may be near universities (student mule recruitment) or in areas with high financial distress.
* **Usage:** Used to build branch-level edges in the transaction graph. Accounts opened at the same fraud-hotspot branch are connected in the GNN graph.

### F3891 — Declared Occupation (String) ⭐ CORE FEATURE

* **Observed range:** 7 categories: "student", "salaried", "selfemployed", "agriculture", "housewife", "retired", "others"
* **Real meaning:** The occupation the customer declared when opening the account
* **Fraud signal:** This is the  **core feature for peer baselines** . Without it, "50 transactions in a week" has no context. With it: a selfemployed person with 50 transactions is normal; a student with 50 transactions is extremely anomalous. Every occupation-conditioned composite feature depends on this.
* **Occupation risk weights for composite features:**
  * student: 1.0 (highest risk weight)
  * housewife: 0.9
  * retired: 0.8
  * agriculture: 0.6
  * salaried: 0.4
  * selfemployed: 0.3
* **Occupation income proxies (₹/year):**
  * student: 50,000
  * housewife: 80,000
  * retired: 200,000
  * agriculture: 150,000
  * salaried: 500,000
  * selfemployed: 600,000

### F3894 — Transaction Count in Observation Window

* **Observed range:** 2–90 integer
* **Real meaning:** Number of transactions in the observation period (likely 30 or 90 days)
* **Fraud signal:** High count on an account with missing tenure (F3043 = NULL) =  **immediate activation pattern** . A ghost account that suddenly has 85 transactions after months of zero activity has been activated for mule use. Used in `occupation_velocity_anomaly` (dividing by occupation peer P95 velocity) — the highest-SHAP composite feature.

---

## SECTION 5 — THE 5-STAGE FEATURE SELECTION PIPELINE

### 5.1 Why We Need Feature Reduction (Critical Concept)

Starting with 3,924 features is a problem because:

1. **Curse of dimensionality** : In high-dimensional spaces, all points become approximately equidistant from each other. Distance-based algorithms (KNN, SVM) break down. Even tree-based models have degraded performance with mostly-noise features.
2. **Overfitting** : More parameters than signal means the model memorizes noise in the training data. It achieves 99% training accuracy but fails on new accounts.
3. **Computational cost** : GNN training scales with feature dimensionality. 3,924 features make training 79× slower than 50 features.
4. **Interpretability** : SHAP explanations across 3,924 features are meaningless. SHAP across 55 meaningful features directly identifies what drove the decision.
5. **Inference speed** : Production systems need sub-100ms latency. 3,924 features in real-time scoring is too slow.

The 18 bank-identified key features are **force-included at every stage** — they can never be statistically eliminated.

### 5.2 Pipeline Overview

```
3,924 features
    │
    ▼ Stage 1: Variance Threshold
~1,800 features (dropped ~2,100 near-constant columns)
    │
    ▼ Stage 2: Missing Value Analysis
~1,200 features (dropped >50% missing, created binary flags)
    │
    ▼ Stage 3: Mutual Information Ranking
150 features (top 150 by MI + 18 forced) [~3-5 min runtime]
    │
    ▼ Stage 4: Correlation Deduplication
~100 features (dropped |r| > 0.95 pairs)
    │
    ▼ Stage 5: SHAP Forward Selection
50 features (top 50 by mean |SHAP| + 18 forced)
    │
    ▼ Add 12 composite features + 2 categoricals
~64 final features
```

### 5.3 Stage 1 — Variance Threshold (3,924 → ~1,800)

**Mathematical definition:**

```
Var(X) = E[(X - μ)²] = (1/n) × Σᵢ (xᵢ - μ)²
```

**Decision rule:** Drop feature if Var(X) < 0.01 *after normalization* (not raw values).

**Why 0.01?** This represents features where less than ~10% of accounts have any deviation from the mode value. These are essentially binary or near-constant flags that carry negligible information. At threshold=0, we'd keep features that are constant except for 1–2 data entry errors. At 0.001, we'd keep columns where 99.9% of values are identical — pure noise.

**Implementation detail:** Applied ONLY to numeric columns. String columns (F3889, F3891) bypass this stage entirely. Bank key features are never dropped regardless of variance.

```python
# From feature_selector.py
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

selector = VarianceThreshold(threshold=0.01)
selector.fit(X[numeric_cols])
kept_numeric = [c for c, keep in zip(numeric_cols, selector.get_support()) if keep]

result = pd.concat([X[kept_numeric], X[cat_cols]], axis=1)
```

### 5.4 Stage 2 — Missing Value Analysis (~1,800 → ~1,200)

Different treatment based on % missing:

| Missing %        | Action                                                               | Rationale                                                                                                                              |
| ---------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| > 50%            | **Drop column**(except bank key features)                      | More than half the data is missing — imputation would be fabricating most of the column                                               |
| 40–50%          | **Create binary flag column** `feature_missing`, then impute | The missingness pattern itself may be informative; keep both signals                                                                   |
| < 40%            | **Impute**(median for numeric, mode for categorical)           | Standard imputation, missingness likely random                                                                                         |
| F3043 (67% null) | **Always create F3043_missing flag**                           | Even though 67% null exceeds our 50% threshold, the bank key feature force-include means we keep it AND the nullness is a fraud signal |

**Why median not mean for imputation?**
Financial data has extreme outliers. A mule account with ₹4.5 crore in F3836 will massively inflate the mean. The median is the 50th percentile value — unaffected by outliers. We prefer it universally for financial features.

```python
# Imputation logic from feature_selector.py
for col in X.select_dtypes(include=[np.number]).columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include=['object']).columns:
    if X[col].isnull().any():
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'UNKNOWN')
```

### 5.5 Stage 3 — Mutual Information Ranking (~1,200 → 150)

**What is Mutual Information?**

Mutual Information (MI) measures  *how much knowing feature X reduces our uncertainty about label Y* . Formally:

```
MI(X; Y) = Σₓ Σᵧ P(x,y) × log[ P(x,y) / (P(x) × P(y)) ]
```

This is equivalent to the **KL divergence** between the joint distribution P(X,Y) and the product of marginals P(X)P(Y). When X and Y are independent, MI = 0. When X perfectly predicts Y, MI = H(Y) (the entropy of Y).

**Why MI over Pearson correlation?**

Pearson correlation only measures *linear* relationships:

```
r = Cov(X,Y) / (σ_X × σ_Y)
```

If a feature has a threshold effect — e.g., `occupation_velocity_anomaly` is near-zero for low velocity but jumps sharply above 2.0 — Pearson correlation would be low even though the feature is highly predictive. Mutual Information captures **any** statistical dependency: linear, non-linear, threshold-based, U-shaped, or any other pattern.

**Implementation:**

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(
    X[numeric_cols].fillna(0),
    y,
    discrete_features=False,
    n_neighbors=5,  # k-NN estimator for continuous features
    random_state=42
)

mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
top_by_mi = mi_series.head(150).index.tolist()
must_include = [c for c in BANK_KEY_FEATURES if c in X.columns]
selected = list(set(top_by_mi) | set(must_include))
```

The `n_neighbors=5` parameter controls the k-NN density estimator used to compute MI for continuous variables. This is based on the Kraskov-Stögbauer-Grassberger estimator.

### 5.6 Stage 4 — Correlation Deduplication (150 → ~100)

**The problem:** Two features with Pearson |r| > 0.95 are measuring almost the same thing. Including both adds noise (they introduce correlated errors) without adding signal (one's information is already captured by the other).

**The solution:** Build the upper triangle of the pairwise correlation matrix. For each pair of features with |r| > 0.95, drop the one with lower MI score from Stage 3.

```python
corr_matrix = X[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
# CRITICAL: Never drop bank key features
to_drop = [c for c in to_drop if c not in BANK_KEY_FEATURES]
X = X.drop(columns=to_drop)
```

**Why |r| > 0.95 and not 0.80 or 0.99?** At 0.80, we'd drop many legitimately different features that happen to be correlated (e.g., two different velocity measures might have r=0.82 but capture different aspects). At 0.99, we'd keep almost-identical features. 0.95 is the standard threshold used in financial ML literature.

### 5.7 Stage 5 — SHAP Forward Selection (~100 → 50)

**Why use SHAP for feature selection instead of just feature importance?**

Traditional XGBoost feature importance has two variants:

* **weight** : Number of times a feature is used in splits → biased toward high-cardinality features
* **gain** : Average information gain when the feature is used → biased toward features used in early splits

**SHAP importance** (mean absolute SHAP value) is computed as:

```
mean |SHAP|(feature i) = (1/N) × Σₙ |φᵢ(xⁿ)|   for n=1 to N accounts
```

This measures the **actual contribution to the output** — features that are used rarely but always critically show up prominently. Features that are used often but have small effects show up with low SHAP importance. It's the most honest measure of feature relevance.

**Process:**

1. Train a fast XGBoost model (100 trees only — for speed, not for final use)
2. Compute SHAP values for all accounts using TreeSHAP (exact, polynomial-time)
3. Compute mean |SHAP| per feature
4. Keep top 50 + all 18 bank key features

```python
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, ...)
model.fit(X_numeric, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_numeric)
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_numeric.columns)
top_shap = mean_abs_shap.sort_values(ascending=False).head(50).index.tolist()
selected = list(set(top_shap) | set(must_include))
```

**Saved to artifacts:** `selected_features.json` stores the exact feature list in training order. At inference time, the scoring pipeline aligns the feature matrix to this exact list — missing features get 0.0, extra features are dropped.

---

## SECTION 6 — THE 12 COMPOSITE FEATURES

### 6.1 Why Composite Features?

The 18 bank features individually are informative. But mule detection requires  **cross-feature reasoning** : a student account is not suspicious because of transaction velocity alone, or income level alone — it's suspicious because the COMBINATION of occupation (student) + velocity (85 txns/week) + amount (₹4.5L) is financially impossible. Composite features encode this cross-feature reasoning as explicit numerical signals.

### 6.2 CRITICAL: Order of Computation

```
⚠️ CRITICAL BUG RISK: Composite features MUST be computed BEFORE label encoding.

Training order (enforced in training/run_all.py):
1. composite_features(X)     ← needs raw string occupation "student", "salaried", etc.
2. fit_encoders(X)           ← converts "student" → 6 (int32)
3. fit_scaler(X)             ← converts all numerics to float via RobustScaler

If you swap steps 1 and 2, occupation is already an integer when composite_features
tries to look up the income proxy for "student". The lookup fails silently and
returns default values, corrupting the entire composite feature computation.

This was explicitly identified and fixed in the codebase.
```

### 6.3 All 12 Composite Features — Complete Reference

#### 1. `occupation_velocity_anomaly` ⭐ HIGHEST SHAP (0.284)

```
Formula: F3894 / occupation_95th_percentile_velocity
```

* Computed by grouping accounts by F3891 (occupation) and computing the 95th percentile of F3894 (transaction count) within each group
* Division normalizes velocity by what's "normal" for that occupation
* A student with F3894=85 against a peer P95 of 12 → score = 7.08 (flagrant anomaly)
* A selfemployed person with F3894=85 against a peer P95 of 80 → score = 1.06 (normal)
* **This is why occupation-specific baselines are Innovation #1 in the project**

```python
occ_95th = df.groupby('F3891')['F3894'].transform(
    lambda x: pd.to_numeric(x, errors='coerce').quantile(0.95)
)
df['occupation_velocity_anomaly'] = (safe('F3894') / (occ_95th.fillna(1.0) + 1e-6))
```

#### 2. `pass_through_score`

```
Formula: F527 × |F2737|
```

* F527 near 1.0 = funds pass through with no net accumulation
* |F2737| high = high balance volatility
* Product amplifies both signals: a true pass-through mule has BOTH high F527 AND high F2737

#### 3. `network_centrality_proxy`

```
Formula: F531 × log(1 + F1692)
```

* F531 = shared counterparties with suspicious accounts (network co-transaction score)
* log(1 + F1692) = log-scaled linked account count (log prevents extreme values from dominating)
* Approximates betweenness centrality in the transaction graph without running a full graph algorithm — critical for fast batch scoring

#### 4. `dormancy_activation_signal`

```
Formula: F3043_missing × F3894
```

* F3043_missing = 1 if account has no recorded tenure (ghost account)
* F3894 = current transaction velocity
* If tenure is missing (ghost) AND velocity is suddenly high = mule activation
* Zero when either: account has recorded tenure OR account has low activity

#### 5. `financial_impossibility_score`

```
Formula: |F3836| / income_proxy[F3891]

income_proxy = {
    'student': 50_000,      # ₹50K/year
    'housewife': 80_000,    # ₹80K/year  
    'retired': 200_000,     # ₹2L/year
    'agriculture': 150_000, # ₹1.5L/year
    'salaried': 500_000,    # ₹5L/year
    'selfemployed': 600_000 # ₹6L/year
}
```

* A student (₹50K proxy) transacting ₹4.5L → score = 9.0
* A salaried person (₹5L proxy) transacting ₹4.5L → score = 0.9
* Flags mathematical impossibility of declared income vs actual activity

#### 6. `peer_deviation_combined`

```
Formula: |F2582| + |F2678|
```

* F2582 = deviation from peer spending amounts (z-score)
* F2678 = deviation from peer transaction timing (z-score)
* Mule activity creates anomalies in BOTH amount AND timing simultaneously (nocturnal high-volume transactions)

#### 7. `structuring_risk`

```
Formula: F2122 × F3894
```

* F2122 = cash deposit frequency (proportion of transactions that are cash)
* F3894 = total transaction count
* High cash deposit proportion × high transaction count = smurfing pattern
* A mule depositing ₹9,900 cash 42 times has both high F2122 and high F3894

#### 8. `cross_border_occupation_risk`

```
Formula: F2082 × risk_weight[F3891]

risk_weight = {
    'student': 1.0,
    'housewife': 0.9,
    'retired': 0.8,
    'agriculture': 0.6,
    'salaried': 0.4,
    'selfemployed': 0.3
}
```

* Weights reflect how implausible international transfers are for each occupation
* Student + any international transfer = highest risk (score = F2082 × 1.0)
* Selfemployed + international transfer = lower risk (score = F2082 × 0.3) — it's plausible they have overseas business

#### 9. `temporal_burst_index`

```
Formula: F3894 / max(F3043, 1)
```

* Transactions per day of account age
* A 5-day-old account with 85 transactions = burst index 17
* A 3-year-old account (≈1,095 days) with 85 transactions = burst index 0.08
* Catches immediate-use accounts — opened and immediately activated as mules

#### 10. `risk_cluster_membership`

* Populated from **Louvain community detection** (runs after graph building)
* Each account receives the fraud rate of its Louvain community as this feature value
* If 80% of your transaction community are fraudsters, this feature = 0.80
* Implements guilt-by-association with statistical backing
* Placeholder (0.0) until graph analysis runs; then overwritten

#### 11. `two_hop_contamination`

* Average GNN risk score of accounts two transaction hops away
* Even if your direct counterparties look clean, their counterparties being high-risk is a contamination signal
* Network-effect fraud propagation: the risk of criminal activity "leaks" through the network
* Placeholder (0.0) until GNN scores are computed; then overwritten

#### 12. `international_flag_occupation`

```
Formula: 1 if (F2082 > 0) AND (F3891 ∈ {'student', 'housewife', 'retired'}) else 0
```

* Hard binary flag for a scenario that is near-impossible to justify legitimately
* A housewife receiving international wire transfers is a FATF ML-04 red flag by definition
* No gradation — either you're in this high-risk category or not
* Directly maps to FATF typology ML-04

---

## SECTION 7 — PREPROCESSING: ENCODING AND SCALING

### 7.1 Label Encoding (Categorical → Integer)

The two string columns (F3889 branch codes, F3891 occupations) must be converted to numbers for the ML models.

```python
# From encoder.py
le = LabelEncoder()
X[col] = le.fit_transform(X[col].astype(str))
X[col] = X[col].astype('int32')  # CRITICAL: must remain int32, NOT float

# Saved to encoders.json:
# {'F3891': {'classes': ['agriculture', 'housewife', 'others', 'retired', 'salaried', 'selfemployed', 'student']}}
# agriculture=0, housewife=1, others=2, retired=3, salaried=4, selfemployed=5, student=6
```

**Why int32 specifically?** CatBoost has a special handling mode where if you pass `cat_features=[]` with string categories, it processes them natively. But our labels are already encoded integers, and RobustScaler would turn them to floats. By keeping them as int32 and NOT passing them as `cat_features` to CatBoost, we treat them as numeric features — which is correct since they've been ordinal-encoded.

**Unseen categories at inference:** If a new branch code appears at inference time that wasn't in training, the encoder maps it to -1 (unknown). This is handled gracefully.

### 7.2 RobustScaler (Scaling)

```
RobustScaler formula: X_scaled = (X - median(X)) / IQR(X)
IQR = Q3 - Q1 (interquartile range)
```

**Why RobustScaler over StandardScaler?**

* StandardScaler: `(X - mean) / std` — both mean and std are severely affected by outliers
* RobustScaler: `(X - median) / IQR` — median and IQR are resistant to outliers

In financial fraud data, mule accounts have extreme values (₹4.5L transaction for a student). StandardScaler would let these outliers pull the scale for the entire feature, making all other values compress toward 0. RobustScaler handles outliers correctly.

**Critical detail:** RobustScaler is NOT applied to the label-encoded categorical columns (F3889, F3891). They're already integers with bounded ranges (0–4 for occupation). Scaling them would convert them from int32 to float64 and change their semantic meaning.

```python
# From scaler.py
encoded_cols = _get_encoded_cols()  # Load from encoders.json
cols_to_scale = [c for c in numeric_cols if c not in encoded_cols]
scaler = RobustScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
# Saved to norm_params.json: center_ and scale_ arrays
```

### 7.3 Feature Alignment at Inference (Critical for Production)

After all preprocessing, the feature matrix must match EXACTLY what the model was trained on:

```python
selected_features = load_selected_features()  # From selected_features.json

# Add any missing features with 0.0
for col in selected_features:
    if col not in X_row.columns:
        X_row[col] = 0.0

# Drop any extra features not in training set
X_row = X_row[[c for c in selected_features if c in X_row.columns]]

# Final safety check: add back any still-missing
for col in selected_features:
    if col not in X_row.columns:
        X_row[col] = 0.0

X_row = X_row[selected_features]  # Enforce exact order
```

Any mismatch between training and inference feature sets causes silent failures — the model receives features in wrong positions and produces garbage predictions. This is one of the most common silent failures in deployed ML systems.

---

## KEY JURY QUESTIONS FOR PART 1

**Q: Why did you choose these specific 18 features as "key"?**
A: We didn't — the bank identified them. They represent the bank's domain expertise about which features are most relevant for mule detection based on their internal AML rules and years of investigation experience. We treat them as oracle knowledge and force-include them at every stage. Interestingly, after running our 5-stage pipeline, most of them end up in the top 55 anyway — validating the bank's expert intuition.

**Q: 3,924 features → 55 features. Couldn't you have just used PCA?**
A: PCA (Principal Component Analysis) creates linear combinations of features. The problem is interpretability: if component 3 is "0.3×F527 + 0.15×F2082 + ... + 0.002×F3847", you can't explain to a compliance officer or a judge why an account was flagged. We need interpretable features that map back to real-world signals (pass-through ratio, velocity anomaly) for SHAP explanations, STR generation, and regulatory audit. PCA destroys this interpretability.

**Q: What is the precision-recall tradeoff and how did you handle it?**
A: Precision and recall move in opposite directions as you change the decision threshold. Lowering the threshold catches more fraud (higher recall) but also flags more legitimate accounts (lower precision). Raising the threshold has the opposite effect. For mule detection, both matter operationally: high recall means fewer laundering operations succeed; high precision means analysts aren't overwhelmed with false alarms. We optimize F1 (harmonic mean) which balances both. Our F1=0.910 with precision=0.940 and recall=0.890 shows excellent balance — we catch nearly 9 in 10 mule accounts while only raising false alarms on 6% of flagged accounts.

**Q: Why do you force-include the 18 key features? What if SHAP or MI says they're irrelevant?**
A: If a feature has low MI score or low SHAP value in our model, there are two possible explanations: (1) it's genuinely not predictive given the other features, or (2) it's being masked by correlated features that were selected instead. The bank identified these 18 through years of investigation and legal proceedings — that's a domain expertise signal we respect over any single statistical criterion. Additionally, for regulatory compliance (PMLA audit), we need to demonstrate that known fraud indicators were considered, even if the model ultimately assigns them low weight. Excluding them could expose the bank to regulatory risk.

---

*Continue to Part 2 → The 6-Layer AI Detection Stack*
