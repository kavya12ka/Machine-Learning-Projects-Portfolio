# Machine Learning Projects Portfolio

A collection of end-to-end data science projects covering regression, classification, and clustering, plus a GenAI project that combines financial statement analysis with retrieval-augmented generation (RAG). Each project is taken from raw data through cleaning, EDA, modeling, evaluation, and (where applicable) deployment.

| Project | Type | Problem | Approach / Best Model | Key Result |
|---|---|---|---|---|
| [House Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/Regression_HousePricePrediction) | Regression | Predict property price from location/property attributes | Lasso Regression (α=0.13) | R² = 0.921 |
| [Used Car Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/RegressionAnalysis_UsedCarPricePrediction) | Regression | Predict the resale price of used cars | See project README | See project README |
| [Breast Cancer Classification](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_BreastCancer) | Classification | Predict malignant vs. benign diagnosis from biopsy features | Random Forest | Recall = 1.00, ROC AUC = 0.9995 |
| [Loan Approval Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_LoanApprovalPrediction) | Classification | Predict whether a loan application is approved | See project README | See project README |
| [Retail Shopper Segmentation](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_RetailShopperSegmentation) | Clustering | Segment customers into behavioral shopper groups | K-Means (K=4) | Silhouette = 0.28, ARI = 1.00 (stable) |
| [Wine Chemical Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_WineChemicalProject) | Clustering | Group wines by their chemical properties | See project README | See project README |
| [AI Fundamental Stock Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/GenAI_FundamentalStockAnalysis) | Financial analysis + GenAI (RAG) | Analyze a company's fundamentals and answer questions about them in plain English | Gemini embeddings + FAISS retrieval + Gemini LLM | 4 fiscal years, 25-document knowledge base |

---

## 1. [House Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/Regression_HousePricePrediction) — Regression

Predicts residential property prices (Lakhs INR) from 25 property, location, and listing
features.

- **Pipeline:** missing-value imputation, text/category cleanup, feature engineering
  (listing-date decomposition), VIF-based multicollinearity checks, a `ColumnTransformer`
  preprocessing pipeline, and six modeling approaches (baseline, simple/multiple linear
  regression, PCA, SelectKBest, Ridge, Lasso, polynomial), each compared on R²/RMSE/MAE.
- **Result:** Multiple linear regression alone reaches R² ≈ 0.92; tuned Lasso (α = 0.13)
  edges it out slightly while zeroing out weak features.
- **Tech:** pandas, scikit-learn, statsmodels (VIF), joblib (deployment)

## 2. [Used Car Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/RegressionAnalysis_UsedCarPricePrediction) — Regression

Predicts the resale price of used cars from their attributes. See the project folder for the
dataset, modeling steps, and results.

## 3. [Breast Cancer Classification](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_BreastCancer) — Classification

Predicts malignant vs. benign diagnosis from 9 biopsy-derived numeric features.

- **Pipeline:** duplicate/leakage checks, class-imbalance analysis (RUS/ROS/SMOTE compared
  head-to-head against no balancing), scaling, and four models (Logistic Regression, KNN,
  Decision Tree, Random Forest) evaluated on accuracy, precision, recall, F1, and ROC AUC.
- **Result:** Random Forest deployed — perfect recall (no missed cancer cases) tied with
  Logistic Regression and KNN, plus the highest ROC AUC (0.9995).
- **Tech:** pandas, scikit-learn, imbalanced-learn, joblib (bundled preprocessing + model
  pipeline)

## 4. [Loan Approval Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_LoanApprovalPrediction) — Classification

Predicts whether a loan application will be approved. See the project folder for the dataset,
models compared, and evaluation results.

## 5. [Retail Shopper Segmentation](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_RetailShopperSegmentation) — Clustering

Segments retail customers into behavioral groups from 13 spend/engagement features.

- **Pipeline:** outlier capping (IQR), multicollinearity-driven feature pruning, elbow
  method + silhouette score for K selection (with an explicit, evidence-based reconciliation
  of the two when they disagreed), K-Means clustering, PCA visualization, and a stability
  check across random seeds.
- **Result:** K=4 selected over the elbow's nominal K=5 after showing K=5 just fragments the
  largest cluster without improving separation; solution is fully stable (ARI = 1.00 across
  seeds).
- **Tech:** pandas, scikit-learn, seaborn/matplotlib

## 6. [Wine Chemical Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_WineChemicalProject) — Clustering

Groups wines by their chemical properties using unsupervised learning. See the project folder
for the dataset, clustering approach, and results.

## 7. [AI Fundamental Stock Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/GenAI_FundamentalStockAnalysis) — Financial Analysis + GenAI (RAG)

Pulls Microsoft (`MSFT`) annual financial statements and 5 years of daily prices from Yahoo
Finance, calculates fundamental and market-risk metrics, and lets you ask questions about the
results in plain English through a small RAG pipeline.

- **Fundamental metrics:** revenue / net income / EPS growth; gross, operating and net margins;
  ROE and ROA; debt-to-equity, current ratio, net cash; operating cash flow, capex, free cash
  flow, and shareholder returns (buybacks + dividends) as a share of FCF.
- **Market metrics:** 52-week high/low, 1/3/5-year returns, annualized return, annualized
  volatility, maximum drawdown, Sharpe ratio, and a valuation snapshot (P/E, price/FCF,
  price/sales).
- **RAG pipeline:** each fiscal year's metrics become short text documents (fiscal year in every
  header) plus multi-year trend documents; they are embedded with `gemini-embedding-001`,
  normalized, and stored in a FAISS index (cosine similarity). The 6 most relevant documents are
  sent to a Gemini model that is instructed to answer only from the retrieved context, cite the
  fiscal year next to every figure, and say so when the data isn't available.
- **Key finding in the data:** revenue and profit keep growing (FY2023 → FY2026), but capex has
  roughly quadrupled, so free cash flow has been falling since FY2024 even as operating cash flow
  rises.
- **Tech:** pandas, numpy, yfinance, FAISS, Google Gemini API (`google-genai`)
- **Note:** needs a free Gemini API key (Colab Secret `GOOGLE_API_KEY`, or the `GEMINI_API_KEY`
  environment variable). Educational analysis only, not financial advice.

---

## Common Structure

Most project folders follow the same layout:

```
<project-folder>/
├── <Notebook_Name>.ipynb   # Full analysis: data → EDA → modeling → evaluation → deployment
├── <model_name>.pkl        # Saved trained model/pipeline (generated by the notebook)
└── README.md               # Project-specific write-up, dataset details, and results table
```

The stock analysis project has a notebook and README only; it has no saved model file because it
calls the Gemini API at run time.

## Skills Demonstrated

- Data cleaning: missing values, duplicates, inconsistent categorical text, mixed-type columns
- Feature engineering and multicollinearity diagnosis (correlation heatmaps, VIF)
- Proper train/test methodology (stratified splits, no leakage into scalers/encoders)
- Model comparison across multiple algorithms per problem type, with metrics chosen to match
  the problem (RMSE/R² for regression; recall/ROC AUC for imbalanced classification;
  silhouette/stability for clustering)
- Hyperparameter tuning (GridSearchCV) and dimensionality reduction (PCA, SelectKBest)
- Model deployment via serialized `scikit-learn` pipelines (`joblib`)
- Financial statement analysis: growth, margins, returns on capital, leverage, free cash flow,
  and market risk metrics (volatility, drawdown, Sharpe ratio)
- Retrieval-augmented generation: document design, embeddings, vector search with FAISS,
  prompt grounding, and handling API rate limits

## Setup

Most notebooks pull their dataset directly from a public URL, so no manual download is needed.
The stock analysis notebook downloads live data from Yahoo Finance and also needs a Gemini API
key.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels imbalanced-learn joblib kneed yfinance faiss-cpu google-genai
```

Never commit API keys to the repository.

## About

This portfolio is a work in progress. Next planned additions are a time-series forecasting
project, multi-company comparison for the stock analysis project, and eventually a small
deployed app (Streamlit/Flask) wrapping one of the models above so it's usable outside a
notebook.
