"""
Customer Purchase Behavior Segmentation
---------------------------------------
Pipeline: load transactions -> build RFM features -> scale -> choose k (elbow +
silhouette) -> KMeans -> label segments -> save results and charts.

Usage:
    python customer_segmentation.py                    # uses generated sample data
    python customer_segmentation.py --data sales.csv   # use your own data
    python customer_segmentation.py --clusters 4       # force number of segments

CSV format (one row per purchase):
    customer_id, order_date, amount
(Online Retail / UCI dataset works after renaming columns:
 CustomerID->customer_id, InvoiceDate->order_date, Quantity*UnitPrice->amount)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = "output"


# ------------------------------------------------------------------ data
def generate_sample_data(n_customers=600, seed=42):
    """Create realistic fake transactions with 4 hidden customer types."""
    rng = np.random.default_rng(seed)
    types = {  # (share, orders, avg_amount, days_since_last_purchase_max)
        "loyal":   (0.15, (15, 30), (80, 150), 30),
        "regular": (0.35, (6, 12),  (40, 80),  90),
        "new":     (0.25, (1, 3),   (30, 70),  45),
        "lapsed":  (0.25, (2, 6),   (20, 60),  360),
    }
    today = pd.Timestamp("2026-10-01")
    rows, cid = [], 1000
    for share, (lo, hi), (a_lo, a_hi), max_recency in types.values():
        for _ in range(int(n_customers * share)):
            cid += 1
            n_orders = rng.integers(lo, hi + 1)
            last = rng.integers(1, max_recency + 1)
            span = 365 if max_recency > 100 else max_recency + 60
            days = np.sort(rng.integers(last, last + span, n_orders))
            avg = rng.uniform(a_lo, a_hi)
            for d in days:
                rows.append((cid, today - pd.Timedelta(days=int(d)),
                             round(max(5, rng.normal(avg, avg * 0.25)), 2)))
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])


def load_data(path=None):
    if path:
        df = pd.read_csv(path, parse_dates=["order_date"])
        df = df[["customer_id", "order_date", "amount"]].dropna()
        return df[df["amount"] > 0]
    print("[info] No dataset supplied -> generating sample transactions.")
    return generate_sample_data()


# ------------------------------------------------------------------ features
def build_rfm(df):
    """Recency (days), Frequency (orders), Monetary (total spend) per customer."""
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("amount", "sum"),
    )
    return rfm


def choose_k(X, k_range=range(2, 9)):
    """Pick k with the best silhouette score; also returns scores for plotting."""
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    best_k = list(k_range)[int(np.argmax(sils))]
    return best_k, list(k_range), inertias, sils


def name_segments(profile, rfm):
    """Name each cluster with simple business rules based on its RFM averages."""
    avg_r, avg_f = rfm["recency"].mean(), rfm["frequency"].mean()
    champion = (profile["frequency"] / avg_f + profile["monetary"] / rfm["monetary"].mean()
                - profile["recency"] / avg_r).idxmax()
    names = {}
    for cl, row in profile.iterrows():
        if cl == champion:
            names[cl] = "Champions"
        elif row["recency"] > 1.5 * avg_r:
            names[cl] = "At Risk / Lapsed" if row["frequency"] >= avg_f else "Lost / Hibernating"
        elif row["frequency"] < 0.5 * avg_f:
            names[cl] = "New Customers"
        else:
            names[cl] = "Loyal Customers"
    # make duplicate names unique
    seen = {}
    for cl, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[cl] = f"{n} ({seen[n]})"
    return names


# ------------------------------------------------------------------ plots
def save_plots(rfm, k_vals, inertias, sils, best_k):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(k_vals, inertias, "o-"); ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
    ax[1].plot(k_vals, sils, "o-", color="green"); ax[1].axvline(best_k, ls="--", c="red")
    ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/choose_k.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 5))
    for seg, g in rfm.groupby("segment"):
        plt.scatter(g["frequency"], g["monetary"], s=18, alpha=0.7, label=seg)
    plt.xlabel("Frequency (orders)"); plt.ylabel("Monetary (total spend)")
    plt.title("Customer Segments"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/segments_scatter.png", dpi=120); plt.close()

    rfm["segment"].value_counts().plot(kind="bar", color="steelblue", figsize=(7, 4))
    plt.title("Customers per Segment"); plt.ylabel("Customers"); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/segment_sizes.png", dpi=120); plt.close()


# ------------------------------------------------------------------ main
def main(data_path=None, n_clusters=None):
    df = load_data(data_path)
    rfm = build_rfm(df)
    print(f"[info] {len(df)} transactions | {len(rfm)} customers")

    # log-transform skewed values, then scale
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))

    best_k, k_vals, inertias, sils = choose_k(X)
    k = n_clusters or best_k
    print(f"[info] Best k by silhouette = {best_k} | using k = {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rfm["cluster"] = km.labels_
    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    names = name_segments(profile, rfm)
    rfm["segment"] = rfm["cluster"].map(names)

    summary = rfm.groupby("segment").agg(
        customers=("cluster", "count"),
        avg_recency_days=("recency", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1).sort_values("total_revenue", ascending=False)
    summary["revenue_share_%"] = (100 * summary["total_revenue"] / summary["total_revenue"].sum()).round(1)

    print("\n=== Segment Summary ===")
    print(summary.to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    rfm.reset_index().to_csv(f"{OUT_DIR}/customer_segments.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/segment_summary.csv")
    save_plots(rfm, k_vals, inertias, sils, best_k)
    print(f"\n[saved] Results and charts in ./{OUT_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Customer Purchase Behavior Segmentation")
    p.add_argument("--data", help="CSV with columns: customer_id, order_date, amount")
    p.add_argument("--clusters", type=int, help="Force number of segments")
    a = p.parse_args()
    main(a.data, a.clusters)
