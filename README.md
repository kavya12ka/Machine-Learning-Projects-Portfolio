# 🤖 Machine Learning Projects Portfolio

End-to-end machine learning projects covering regression, classification, clustering, and generative AI, each taken from raw data through cleaning, EDA, modeling, and evaluation.

---

## 📈 [AI Fundamental Stock Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/GenAI_FundamentalStockAnalysis)

Analyzed four fiscal years of Microsoft financial statements and five years of price history (growth, margins, ROE/ROA, free cash flow, volatility, drawdown, Sharpe ratio), then built a RAG question-answering system on Gemini embeddings and FAISS — revealing capex rising about 4x from FY2023 to FY2026 while free cash flow fell after FY2024. **Tools:** Python, Pandas, yfinance, FAISS, Gemini API

## 🏠 [House Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/Regression_HousePricePrediction)

Predicted property prices (Lakhs INR) from 25 property, location, and listing features, checking multicollinearity with VIF and comparing six modeling approaches — with a tuned Lasso (α = 0.13) reaching R² = 0.921. **Tools:** Python, Pandas, scikit-learn, statsmodels, joblib

## 🚗 [Used Car Price Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/RegressionAnalysis_UsedCarPricePrediction)

Built and evaluated regression models to predict the resale price of used cars from their attributes, from data cleaning and EDA through model comparison. **Tools:** Python, Pandas, scikit-learn

## 🩺 [Breast Cancer Classification](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_BreastCancer)

Classified tumors as malignant or benign from 9 biopsy features, comparing RUS, ROS, and SMOTE balancing and four models — Random Forest was deployed with perfect recall (no missed cancer cases) and a ROC AUC of 0.9995. **Tools:** Python, Pandas, scikit-learn, imbalanced-learn, joblib

## 🏦 [Loan Approval Prediction](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClassificationAnalysis_LoanApprovalPrediction)

Built and evaluated classification models to predict whether a loan application is approved, covering data cleaning, EDA, and model comparison. **Tools:** Python, Pandas, scikit-learn

## 🛍️ [Retail Shopper Segmentation](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_RetailShopperSegmentation)

Segmented customers into behavioral groups from 13 spend and engagement features, using the elbow method and silhouette score to pick K = 4 over K = 5 — a solution that stayed fully stable across random seeds (ARI = 1.00). **Tools:** Python, Pandas, scikit-learn, Matplotlib, Seaborn

## 🍷 [Wine Chemical Analysis](https://github.com/kavya12ka/Machine-Learning-Projects-Portfolio/tree/main/ClusteringAnalysis_WineChemicalProject)

Applied unsupervised learning to group wines by their chemical properties, from preprocessing and feature scaling through clustering and interpretation. **Tools:** Python, Pandas, scikit-learn

---

## 🛠 Overall Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `statsmodels` · `imbalanced-learn` · `FAISS` · `Gemini API` · `yfinance` · `joblib` · `Jupyter Notebook`
