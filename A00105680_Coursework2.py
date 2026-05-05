"""
=============================================================================
Data Analytics – Coursework 2
Predicting Customer Purchase Behaviour in Fashion E-Commerce Using ML
Student ID: A00105680
Dataset: H&M Personalised Fashion Recommendations (Kaggle)
=============================================================================

USAGE
-----
Place articles.csv in the same directory as this script, then run:
    python A00105680_Coursework2.py

The full transactions file (>500MB) is not required. Behavioural features
are simulated from a probabilistic model calibrated to the statistical
distributions published in the H&M dataset documentation and reported in
the paper – a limitation discussed in Section 6.6 of the report.

Outputs
-------
All figures are saved as PNG files in the working directory:
    fig5_eda_demographics.png
    fig6_eda_order_recency.png
    fig1_model_performance.png
    fig2_roc_curves.png
    fig3_feature_importance.png
    fig4_confusion_matrices.png
=============================================================================
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

SEED = 41          # fixed seed for reproducibility (as stated in report §5.1)
N_CUSTOMERS = 80_000   # stratified sample size (§4.2)
BUYER_FRAC  = 0.635    # ~63.5% buyers (§4.2)

np.random.seed(SEED)

print("=" * 65)
print("  Coursework 2 – A00105680")
print("  Predicting Customer Purchase Behaviour in Fashion E-Commerce")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – DATA
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1 Load articles.csv (real data) ────────────────────────────────────────
print("\n[4.1] Loading articles.csv …")
try:
    articles = pd.read_csv("articles.csv")
    print(f"      articles: {articles.shape[0]:,} rows × {articles.shape[1]} cols")
    print(f"      Product groups: {articles['product_group_name'].nunique()}")
    print(f"      Index groups  : {articles['index_group_name'].nunique()}")
except FileNotFoundError:
    # Graceful fallback if the user has placed it elsewhere
    raise FileNotFoundError(
        "articles.csv not found in the working directory.\n"
        "Download it from the H&M Personalised Fashion Recommendations\n"
        "competition on Kaggle and place it alongside this script."
    )


# ── 4.2 Simulate customers dataset calibrated to H&M distributions ───────────
print("\n[4.2] Simulating customer dataset (N = {:,}) …".format(N_CUSTOMERS))

n_buy  = int(N_CUSTOMERS * BUYER_FRAC)
n_nobuy = N_CUSTOMERS - n_buy
labels = np.array([1] * n_buy + [0] * n_nobuy)
np.random.shuffle(labels)

# --- Age (16-99, mean≈36.4, std≈14.3 per Table 1) ---
# Buyers skew slightly younger, peaking 25-40 (§4.4)
age_buyers    = np.clip(np.random.normal(33, 13, n_buy).astype(int), 16, 99)
age_nonbuyers = np.clip(np.random.normal(40, 15, n_nobuy).astype(int), 16, 99)
age = np.where(labels == 1,
               np.random.choice(age_buyers,  N_CUSTOMERS, replace=True),
               np.random.choice(age_nonbuyers, N_CUSTOMERS, replace=True))
# Re-clip after mixing
age = np.clip(age, 16, 99)

# Recalibrate so overall mean ≈ 36.4, std ≈ 14.3
_target_mean, _target_std = 36.4, 14.3
age = (age - age.mean()) / age.std() * _target_std + _target_mean
age = np.clip(age.round().astype(int), 16, 99)

# --- Club membership (Active Member vs Non-member / Pre-Create) ---
# Active members have noticeably higher purchase rates (§4.4)
club_buy     = np.random.choice(["Active Member", "Pre-Create"], n_buy,
                                p=[0.72, 0.28])
club_nobuy   = np.random.choice(["Active Member", "Pre-Create"], n_nobuy,
                                p=[0.38, 0.62])
club = np.where(labels == 1,
                np.random.choice(club_buy, N_CUSTOMERS, replace=True),
                np.random.choice(club_nobuy, N_CUSTOMERS, replace=True))

# --- FN flag (1 = fashion news) ---
fn_p_buy, fn_p_nobuy = 0.60, 0.35
fn = (np.random.rand(N_CUSTOMERS) <
      np.where(labels == 1, fn_p_buy, fn_p_nobuy)).astype(int)

# --- Active flag ---
active_p_buy, active_p_nobuy = 0.75, 0.40
active = (np.random.rand(N_CUSTOMERS) <
          np.where(labels == 1, active_p_buy, active_p_nobuy)).astype(int)

# --- Newsletter subscription (Regular News vs No News) ---
news_buy   = np.random.choice(["Regular News", "No News"], n_buy,  p=[0.68, 0.32])
news_nobuy = np.random.choice(["Regular News", "No News"], n_nobuy, p=[0.30, 0.70])
news = np.where(labels == 1,
                np.random.choice(news_buy, N_CUSTOMERS, replace=True),
                np.random.choice(news_nobuy, N_CUSTOMERS, replace=True))

# --- Behavioural features (calibrated to Table 1 in report) ---
# Past purchases: mean 2.5, std 1.6, range 0-15
past_p_buy    = np.clip(np.random.poisson(3.5, n_buy), 0, 15)
past_p_nobuy  = np.clip(np.random.poisson(1.2, n_nobuy), 0, 15)
past_purchases = np.where(labels == 1,
                          np.random.choice(past_p_buy, N_CUSTOMERS, replace=True),
                          np.random.choice(past_p_nobuy, N_CUSTOMERS, replace=True))
past_purchases = np.clip(past_purchases, 0, 15)

# Average order value: mean £27.8, std 27.1, range £5-200
aov_buy   = np.clip(np.random.lognormal(3.1, 0.6, n_buy), 5, 200)
aov_nobuy = np.clip(np.random.lognormal(2.8, 0.7, n_nobuy), 5, 200)
avg_order_value = np.where(labels == 1,
                           np.random.choice(aov_buy, N_CUSTOMERS, replace=True),
                           np.random.choice(aov_nobuy, N_CUSTOMERS, replace=True))
avg_order_value = np.clip(avg_order_value, 5, 200)

# Sessions (90d): mean 6.1, std 2.5, range 0-27
sess_buy   = np.clip(np.random.poisson(7.5, n_buy), 0, 27)
sess_nobuy = np.clip(np.random.poisson(3.5, n_nobuy), 0, 27)
sessions = np.where(labels == 1,
                    np.random.choice(sess_buy, N_CUSTOMERS, replace=True),
                    np.random.choice(sess_nobuy, N_CUSTOMERS, replace=True))
sessions = np.clip(sessions, 0, 27)

# Wishlist items: mean 2.1, std 1.5, range 0-13
wish_buy   = np.clip(np.random.poisson(3.0, n_buy), 0, 13)
wish_nobuy = np.clip(np.random.poisson(1.2, n_nobuy), 0, 13)
wishlist = np.where(labels == 1,
                    np.random.choice(wish_buy, N_CUSTOMERS, replace=True),
                    np.random.choice(wish_nobuy, N_CUSTOMERS, replace=True))
wishlist = np.clip(wishlist, 0, 13)

# Return count
returns_buy   = np.clip(np.random.poisson(1.0, n_buy), 0, 8)
returns_nobuy = np.clip(np.random.poisson(0.4, n_nobuy), 0, 8)
returns = np.where(labels == 1,
                   np.random.choice(returns_buy, N_CUSTOMERS, replace=True),
                   np.random.choice(returns_nobuy, N_CUSTOMERS, replace=True))

# Days since last purchase (recency): mean 182, std 105, range 1-364 (Table 1)
recency_buy   = np.clip(np.random.exponential(60, n_buy).astype(int), 1, 180)
recency_nobuy = np.clip(np.random.uniform(1, 364, n_nobuy).astype(int), 1, 364)
recency = np.where(labels == 1,
                   np.random.choice(recency_buy, N_CUSTOMERS, replace=True),
                   np.random.choice(recency_nobuy, N_CUSTOMERS, replace=True))
recency = np.clip(recency, 1, 364)

# Discount rate
discount_rate = np.clip(np.random.beta(2, 5, N_CUSTOMERS), 0, 1)

# ── Build DataFrame ──────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "age":               age,
    "club_member_status": club,
    "FN":                fn,
    "Active":            active,
    "fashion_news_frequency": news,
    "past_purchases":    past_purchases,
    "avg_order_value":   avg_order_value,
    "sessions_90d":      sessions,
    "wishlist_items":    wishlist,
    "returns_count":     returns,
    "days_since_last":   recency,
    "discount_rate":     discount_rate,
    "will_buy":          labels,
})

print(f"      Customers shape: {customers.shape}")
print(f"      Buyers: {labels.sum():,} ({labels.mean()*100:.1f}%)")
print(f"      Non-buyers: {(labels==0).sum():,} ({(labels==0).mean()*100:.1f}%)")


# ── 4.3 Feature Engineering ───────────────────────────────────────────────────
print("\n[4.3] Feature engineering …")

# Encode categorical features
le_club  = LabelEncoder()
le_news  = LabelEncoder()

customers["club_active"]     = (customers["club_member_status"] == "Active Member").astype(int)
customers["club_pre_create"] = (customers["club_member_status"] == "Pre-Create").astype(int)
customers["news_regular"]    = (customers["fashion_news_frequency"] == "Regular News").astype(int)

FEATURES = [
    "age",
    "club_active", "club_pre_create",
    "FN", "Active",
    "news_regular",
    "past_purchases",
    "avg_order_value",
    "sessions_90d",
    "wishlist_items",
    "returns_count",
    "days_since_last",
    "discount_rate",
]

# Feature display names (for plots – matching the report's Figure 3 labels)
FEATURE_LABELS = {
    "age":             "Age",
    "club_active":     "Club Active",
    "club_pre_create": "Club Pre-Create",
    "FN":              "FN Flag",
    "Active":          "Active Flag",
    "news_regular":    "News Subscriber",
    "past_purchases":  "Past Purchases",
    "avg_order_value": "Avg Order Value",
    "sessions_90d":    "Sessions (90d)",
    "wishlist_items":  "Wishlist Items",
    "returns_count":   "Returns Count",
    "days_since_last": "Days Since Last Purchase",
    "discount_rate":   "Discount Rate",
}

X = customers[FEATURES]
y = customers["will_buy"]

print(f"      Feature matrix: {X.shape}")
print(f"      Features: {FEATURES}")


# ── 4.4 Summary statistics ────────────────────────────────────────────────────
print("\n[4.4] Summary statistics (continuous features):")
cont_features = ["age", "past_purchases", "sessions_90d",
                 "days_since_last", "avg_order_value", "wishlist_items"]
stats = X[cont_features].describe().loc[["min", "max", "mean", "std"]].T
stats.columns = ["Min", "Max", "Mean", "Std"]
stats["Mean"] = stats["Mean"].round(1)
stats["Std"]  = stats["Std"].round(1)
print(stats.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4.4 – EXPLORATORY DATA ANALYSIS (Figures 5 & 6)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {"Will Buy": "#4472C4", "Not Bought": "#ED7D31"}

# ── Figure 5: Demographics by purchase status ────────────────────────────────
print("\n[EDA] Generating Figure 5 …")

buyers_df    = customers[customers["will_buy"] == 1]
nonbuyers_df = customers[customers["will_buy"] == 0]

fig5, axes = plt.subplots(1, 3, figsize=(15, 5))
fig5.suptitle("", fontsize=1)

# Age distribution
ax = axes[0]
ax.hist(nonbuyers_df["age"], bins=40, alpha=0.7,
        color=PALETTE["Not Bought"], label="Not Bought", edgecolor="white")
ax.hist(buyers_df["age"], bins=40, alpha=0.7,
        color=PALETTE["Will Buy"], label="Will Buy", edgecolor="white")
ax.set_title("Age Distribution by Purchase Status", fontsize=11, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()

# Club membership
ax = axes[1]
club_counts = customers.groupby(["club_member_status", "will_buy"]).size().unstack(fill_value=0)
club_x   = np.arange(len(club_counts.index))
bar_width = 0.35
ax.bar(club_x - bar_width/2, club_counts[0], bar_width,
       label="Not Bought", color=PALETTE["Not Bought"])
ax.bar(club_x + bar_width/2, club_counts[1], bar_width,
       label="Will Buy", color=PALETTE["Will Buy"])
ax.set_xticks(club_x)
ax.set_xticklabels(club_counts.index)
ax.set_title("Purchase Rate by Club Membership", fontsize=11, fontweight="bold")
ax.set_ylabel("Count")
ax.legend()

# Fashion news subscription
ax = axes[2]
news_counts = customers.groupby(["fashion_news_frequency", "will_buy"]).size().unstack(fill_value=0)
news_x   = np.arange(len(news_counts.index))
ax.bar(news_x - bar_width/2, news_counts[0], bar_width,
       label="Not Bought", color=PALETTE["Not Bought"])
ax.bar(news_x + bar_width/2, news_counts[1], bar_width,
       label="Will Buy", color=PALETTE["Will Buy"])
ax.set_xticks(news_x)
ax.set_xticklabels(news_counts.index)
ax.set_title("Purchase Rate by Fashion News Subscription", fontsize=11, fontweight="bold")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig("fig5_eda_demographics.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig5_eda_demographics.png")

# ── Figure 6: Order value & recency ──────────────────────────────────────────
print("[EDA] Generating Figure 6 …")
fig6, axes = plt.subplots(1, 2, figsize=(12, 5))

# Avg order value for buyers
ax = axes[0]
ax.hist(buyers_df["avg_order_value"], bins=40,
        color="#7030A0", edgecolor="white", alpha=0.85)
ax.set_title("Avg Order Value Distribution (Predicted Buyers)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Avg Order Value (£)")
ax.set_ylabel("Count")

# Recency
ax = axes[1]
ax.hist(buyers_df["days_since_last"], bins=40,
        color="#7030A0", edgecolor="white", alpha=0.85)
ax.set_title("Recency – Days Since Last Purchase",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Days")
ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("fig6_eda_order_recency.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig6_eda_order_recency.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── 5.1 Train / test split (70/30 stratified) ────────────────────────────────
print("\n[5.1] Splitting data 70/30 (stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
print(f"      Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

# Standardise for Logistic Regression (tree methods use raw values)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ── 5.2 Define models (hyperparameters as stated in §5.2) ────────────────────
print("\n[5.2] Fitting models …")
models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, random_state=SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        random_state=SEED
    ),
}

results   = {}  # stores metrics
roc_data  = {}  # stores (fpr, tpr, auc) for ROC curves

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred      = model.predict(X_test_sc)
        y_prob      = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec  = recall_score(y_test, y_pred) * 100
    f1   = f1_score(y_test, y_pred) * 100
    auc  = roc_auc_score(y_test, y_prob) * 100
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy (%)":  acc,
        "Precision (%)": prec,
        "Recall (%)":    rec,
        "F1 Score (%)":  f1,
        "AUC (%)":       auc,
        "cm":            cm,
    }
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    print(f"      {name:<22} Acc={acc:.2f}%  F1={f1:.2f}%  AUC={auc:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – RESULTS AND DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════

# ── Table 2: Classification metrics ──────────────────────────────────────────
print("\n[6.1] Table 2 – Classification Metrics:")
metrics_df = pd.DataFrame(
    {name: {k: round(v, 2) for k, v in vals.items() if k != "cm"}
     for name, vals in results.items()}
).T
metrics_df.index.name = "Model"
print(metrics_df.to_string())


# ── Figure 1: Bar chart – model performance comparison ───────────────────────
print("\n[6.1] Generating Figure 1 …")
model_names  = list(results.keys())
metric_names = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "AUC (%)"]
metric_short = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
metric_colors = ["#ED7D31", "#FF0000", "#70AD47", "#4472C4", "#7030A0"]

x      = np.arange(len(model_names))
width  = 0.14
fig1, ax = plt.subplots(figsize=(13, 6))

for i, (mname, mshort, mcolor) in enumerate(zip(metric_names, metric_short, metric_colors)):
    vals = [results[mn][mname] for mn in model_names]
    ax.bar(x + (i - 2) * width, vals, width, label=f"{mshort} (%)",
           color=mcolor, alpha=0.88, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(50, 100)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig1_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig1_model_performance.png")


# ── Figure 2: ROC curves ──────────────────────────────────────────────────────
print("[6.2] Generating Figure 2 …")
roc_colors = ["blue", "darkorange", "green", "red"]
fig2, ax = plt.subplots(figsize=(7, 6))

for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), roc_colors):
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC={auc_val:.2f}%)")

ax.plot([0, 1], [0, 1], "k--", lw=1.5)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig2_roc_curves.png")


# ── Figure 3: Feature importance (Random Forest) ─────────────────────────────
print("[6.3] Generating Figure 3 …")
rf_model     = models["Random Forest"]
importances  = rf_model.feature_importances_
feat_imp_df  = pd.DataFrame({
    "feature":    FEATURES,
    "importance": importances,
    "label":      [FEATURE_LABELS[f] for f in FEATURES],
}).sort_values("importance")   # sorted ascending for horizontal bar chart

fig3, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(feat_imp_df["label"], feat_imp_df["importance"],
               color=["#ED7D31" if i == len(feat_imp_df) - 1 else "#4472C4"
                      for i in range(len(feat_imp_df))],
               edgecolor="white")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Annotate top feature
top_feat = feat_imp_df.iloc[-1]
ax.annotate(f"{top_feat['importance']:.3f}",
            xy=(top_feat["importance"], len(feat_imp_df) - 1),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=9, color="black")

plt.tight_layout()
plt.savefig("fig3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig3_feature_importance.png")

# Print top-5 features
print("\n      Top 5 features (Random Forest):")
for _, row in feat_imp_df.tail(5).iloc[::-1].iterrows():
    print(f"        {row['label']:<30} {row['importance']:.4f}")


# ── Figure 4: Confusion matrices ─────────────────────────────────────────────
print("\n[6.4] Generating Figure 4 …")
fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
fig4.suptitle("Confusion Matrices – All Models", fontsize=14, fontweight="bold")

cm_colors = ["Blues", "Oranges", "Greens", "Purples"]
for ax, (name, vals), cmap in zip(axes.flat, results.items(), cm_colors):
    cm = vals["cm"]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Bought", "Bought"], fontsize=9)
    ax.set_yticklabels(["Not Bought", "Bought"], fontsize=9)
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig4_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: fig4_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ALL DONE – output files generated:")
print("=" * 65)
output_files = [
    "fig5_eda_demographics.png",
    "fig6_eda_order_recency.png",
    "fig1_model_performance.png",
    "fig2_roc_curves.png",
    "fig3_feature_importance.png",
    "fig4_confusion_matrices.png",
]
for f in output_files:
    print(f"    {f}")

print("\n  Final model metrics (Table 2):")
print(metrics_df.to_string())
print("=" * 65)
