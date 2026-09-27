# Breast Cancer Classification Analysis

A supervised machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Breast Cancer diagnostic dataset. The notebook covers the full pipeline: data cleaning, multicollinearity reduction, class balancing, model training/comparison across four algorithms, and deployment of the best-performing model.

## Dataset

- **Source:** [`breast-cancer.csv`](https://raw.githubusercontent.com/kavya12ka/Machine-Learning-Projects-Portfolio/refs/heads/main/ClassificationAnalysis_BreastCancer/breast-cancer.csv)
- **Records:** 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and "worst" value)
- **Target:** `diagnosis` — Malignant (`M` → 1) or Benign (`B` → 0)
- No missing values or duplicate rows.

## Workflow

1. **Data Cleaning**
   - Encoded `diagnosis` as binary (1 = Malignant, 0 = Benign)
   - Dropped the non-predictive `id` column
   - Checked for nulls and duplicates

2. **Multicollinearity Reduction (VIF)**
   - Computed Variance Inflation Factor for all features and iteratively removed the worst offender until all remaining features had VIF ≤ 10
   - Dropped 13 of 30 features (e.g. `radius_mean`, `radius_worst`, `perimeter_mean` had VIF in the hundreds/thousands)
   - Retained **17 features** for modeling

3. **Train/Test Split**
   - 80/20 split, stratified on `diagnosis` (`random_state=42`)

4. **Class Imbalance Handling**
   - Training set was imbalanced toward benign cases
   - Compared three resampling strategies: Random Over Sampler, Random Under Sampler, and **SMOTE** (used for final models)

5. **Feature Scaling**
   - `StandardScaler` fit on the SMOTE-balanced training data, applied to both train and test sets

6. **Modeling**
   Four classifiers were trained and evaluated on the held-out test set:
   - Logistic Regression
   - K-Nearest Neighbors (k=5)
   - Decision Tree
   - Random Forest

7. **Deployment**
   - Best model (Random Forest) and the fitted scaler are serialized with `joblib`
   - Includes a sample inference cell that loads the saved artifacts and predicts on a new patient record

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.956 | 0.951 | 0.929 | 0.940 | 0.993 |
| KNN | 0.921 | 0.902 | 0.881 | 0.892 | 0.984 |
| Decision Tree | 0.912 | 0.921 | 0.833 | 0.875 | 0.896 |
| **Random Forest** | **0.956** | **0.974** | 0.905 | 0.938 | **0.996** |

**Random Forest** was selected as the deployed model, offering the best ROC AUC and precision, with strong overall accuracy. In a diagnostic context, recall (catching malignant cases) is also worth close attention — Logistic Regression edges out slightly on recall, so it's a reasonable alternative depending on the cost trade-off between false positives and false negatives.

## Repository Structure

```
.
├── Classification_CancerAnalysis.ipynb   # Main analysis notebook
├── cancer_classification_model.pkl       # Serialized Random Forest model (generated on run)
├── scaler.pkl                            # Serialized StandardScaler (generated on run)
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
imbalanced-learn
joblib
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels imbalanced-learn joblib
```

## Usage

1. Open `Classification_CancerAnalysis.ipynb` in Jupyter.
2. Run all cells to reproduce data prep, VIF filtering, balancing, model training, and evaluation.
3. The notebook saves `cancer_classification_model.pkl` and `scaler.pkl` at the deployment step.
4. To predict on a new patient, load the artifacts and pass a feature dict matching the 17 retained columns (see the example cell at the end of the notebook):

```python
import joblib
import pandas as pd

model = joblib.load("cancer_classification_model.pkl")
scaler = joblib.load("scaler.pkl")

new_patient = pd.DataFrame([{...}])  # must match the 17 retained feature columns
new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)[:, 1]
```

## Notes & Caveats

- This is an educational/portfolio project, **not a clinical diagnostic tool**.
- Class balancing (SMOTE) was applied only to training data to avoid leaking synthetic samples into the test set.
- Feature selection via VIF addresses multicollinearity for interpretability but was not separately validated against a held-out feature-importance benchmark.
