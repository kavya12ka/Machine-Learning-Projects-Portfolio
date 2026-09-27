# Retail Shopper Segmentation — Clustering Analysis

Unsupervised segmentation of retail customers into distinct shopper types using K-Means,
based on spending, purchase frequency, engagement, and loyalty behavior — with a full
methodology check (outlier handling, multicollinearity, K selection, and cluster stability).

## Dataset

- **Source:** [`retail_customer_clustering.csv`](https://raw.githubusercontent.com/swapnilsaurav/AIML/refs/heads/main/ML/Clustering-Retail-Customers/retail_customer_clustering.csv)
- **Size:** 2,000 customers × 19 columns (no missing values, no duplicates)
- **Features used for clustering** (13, after dropping one redundant feature): `Age`,
  `Annual_Income_INR`, `Total_Spend_12M_INR`, `Purchase_Frequency_12M`, `Avg_Order_Value_INR`,
  `Days_Since_Last_Purchase`, `Discount_Usage_Pct`, `Online_Purchase_Pct`,
  `App_Visits_Per_Month`, `Returns_Pct`, `Categories_Purchased`, `Customer_Tenure_Months`,
  `Campaign_Response_Rate_Pct`
- **Dropped from clustering, used only for profiling:** `Customer_ID`, `Gender`, `City`,
  `Preferred_Channel`, `Signup_Date`

## Repository Structure

```
.
├── Clustering_Analysis_shoppers.ipynb   # Full analysis, from raw data to segment profiles
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
kneed
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

## Usage

Open `Clustering_Analysis_shoppers.ipynb` in Jupyter and run all cells top to bottom — the
dataset is pulled directly from the URL above, no manual download needed. The notebook ends
with each customer labeled by cluster (`df1['Clusters']`), which can be joined back to the
original customer table for downstream use (targeting, CRM segments, etc.).

## Pipeline Overview

1. **Data cleaning** — verified no missing values or duplicate rows (dataset was already
   clean); dropped identifier/categorical columns (`Customer_ID`, `Gender`, `City`,
   `Preferred_Channel`, `Signup_Date`) from the clustering feature set, keeping them aside
   for profiling afterward.
2. **Outlier treatment** — capped extreme values per feature using the IQR rule, since
   K-Means is distance-based and sensitive to outliers.
3. **Multicollinearity check** — a correlation heatmap flagged `Loyalty_Points` as
   redundant with `Total_Spend_12M_INR` (r ≈ 0.91); dropped to avoid double-weighting that
   signal in the distance calculation.
4. **Preprocessing** — median imputation, then `StandardScaler`.
5. **Choosing K** — computed both the elbow method (inertia) and silhouette score across
   K=2–10. The elbow curve nominally suggested K≈5, but an automated knee detector and a
   direct K=4-vs-K=5 comparison showed K=5 mostly just splits the largest K=4 cluster into
   two weaker, less-separated halves — so **K=4** was chosen, backed by the highest
   silhouette score (0.295) and full stability across random seeds (ARI = 1.000).
6. **Clustering** — K-Means (K=4) on the scaled, cleaned feature set.
7. **Dimensionality reduction for visualization** — PCA to 2 components for a 2D scatter
   of the clusters (captures a partial, not complete, view of the separation).
8. **Cluster profiling** — per-cluster feature means and categorical breakdowns
   (`Gender`, `City`, `Preferred_Channel`) to turn cluster numbers into shopper descriptions.
9. **Stability validation** — re-ran K-Means with 5 different random seeds and compared
   labelings via Adjusted Rand Index (ARI = 1.000 across all seeds — fully stable).

## Results

**Cluster sizes:** 960 / 416 / 255 / 369 (out of 2,000 customers)

| Segment | Size | Age | Income (INR) | Annual Spend (INR) | Frequency/yr | Avg Order Value | Discount Usage | Tenure (mo.) | Campaign Response |
|---|---|---|---|---|---|---|---|---|---|
| **0 — Engaged Discount Shoppers** | 960 | 32.0 | ₹9.8L | ₹61.5K | 20.0 | ₹3,340 | 53.9% | 34.3 | 58.4% |
| **1 — Dormant / At-Risk** | 416 | 40.2 | ₹9.2L | ₹22.5K | 5.0 | ₹5,075 | 45.8% | 44.2 | 12.4% |
| **2 — Premium Infrequent Spenders** | 255 | 47.2 | ₹22.1L | ₹108.2K | 6.0 | ₹10,965 | 19.2% | 55.4 | 29.7% |
| **3 — High-Value Loyalists** | 369 | 43.3 | ₹17.7L | ₹123.9K | 26.8 | ₹4,783 | 12.5% | 62.8 | 53.9% |

**Segment descriptions:**

- **Segment 0 — Engaged Discount Shoppers (48% of customers):** Younger, buy often (20x/yr),
  purchase mostly online/via app, very recently active (18 days since last purchase), and
  lean heavily on discounts (53.9% usage) — the volume-driven, price-sensitive segment.
- **Segment 1 — Dormant / At-Risk (21%):** Lowest spend and frequency, longest gap since
  last purchase (111 days), barely uses the app, and has by far the weakest campaign
  response (12.4%) — the segment most at risk of churn.
- **Segment 2 — Premium Infrequent Spenders (13%, smallest):** Highest income and by far
  the highest average order value (₹10,965 vs. ₹3–5K elsewhere), but buys rarely (6x/yr)
  and barely uses discounts — a big-ticket, low-touch luxury buyer.
- **Segment 3 — High-Value Loyalists (18%):** Highest total annual spend, highest purchase
  frequency, longest tenure (62.8 months), and strong campaign response, while barely using
  discounts — the most valuable, loyal, full-price segment.

## Key Findings

- The elbow method's usual visual read (K≈5) doesn't hold up under closer inspection — the
  extra cluster it implies doesn't reveal a genuinely new shopper type, it just splits an
  existing large segment into two similar, weaker halves.
- The chosen 4-segment solution is **fully stable**: identical clustering across 5 different
  random initializations (ARI = 1.000), so the segments aren't an artifact of a lucky seed.
- Segments differ far more on **behavior** (frequency, discount reliance, recency, campaign
  response) than on **demographics** — gender split is roughly even across all four segments.
- The two "high spend" segments (2 and 3) reach similar annual spend through opposite
  behavior: Segment 2 spends big rarely (premium, low-touch), Segment 3 spends often at
  moderate order sizes (loyal, frequent) — worth targeting with very different strategies.
- Segment 1 (dormant/at-risk) is a clear re-engagement target: low everything except tenure,
  and the weakest response to campaigns run so far.

## License

Add a license of your choice (e.g., MIT) if you plan to make this repository public.
