# 🧠 Obesity Risk Prediction using ML

A complete machine learning pipeline to predict obesity levels based on lifestyle, dietary, and physical attributes. This project covers data preprocessing, exploratory data analysis (EDA), model building, and evaluation to derive insights on the relationship between daily habits and obesity levels.

---

## 📁 Project Structure

```
├── data/                   # Train & Test datasets
├── notebooks/              # EDA & Model experiments
├── models/                 # Saved models
├── utils/                  # Helper functions for preprocessing, visualization, etc.
├── obesity_prediction.py   # Main pipeline script
└── README.md               # Project overview
```

---

## 🔎 PART ONE: Data Preprocessing

### 📌 Null Value Treatment

- **Train Set:**
  - `FCVC` (12 nulls): Median imputation (ordinal numerical)
  - `CALC` (28 nulls): Mode imputation (categorical)
- **Test Set:** No null values

### 🔤 Categorical Variables & Encoding

- **Binary Nominal** (`family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC`) → *Label Encoding*
- **Ordinal** (`CAEC`, `CALC`) → *Ordinal Encoding*
- **Nominal**:
  - `Gender` → *Label Encoding*
  - `MTRANS` → *One-Hot Encoding*
- **Target** (`NObeyesdad`) → *Ordinal Encoding*

### 📏 Feature Scaling

- **Numerical Columns**:
  ```
  ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
  ```
- **Scaling Strategy**:
  - `RobustScaler`: `Age`, `Weight` (due to outliers)
  - `StandardScaler`: Remaining numerical features

### 🎯 Feature Selection

- **Filter Method (Pearson Correlation)**:
  - Dropped:
    ```
    FCVC, CALC, CH2O, Height, NCP, SMOKE, Gender, TUE, SCC, MTRANS_*
    ```

- **Wrapper Method (Random Forest)**:
  - Top features:
    ```
    Weight, Height, FCVC, Age, Gender, NCP, TUE, CH2O, FAF, CAEC
    ```

- **Final Dropped Features** (intersection):
  ```
  SMOKE, CALC, MTRANS_Bike, MTRANS_Motorbike, MTRANS_Public_Transportation, MTRANS_Walking
  ```

### 🧮 Feature Engineering

- **BMI** = `Weight / (Height²)`
- Correlations with target:
  - Pearson: `0.23`
  - Spearman: `0.84` → strong nonlinear relationship

---

## 📊 PART TWO: Exploratory Data Analysis (EDA)

### 🧰 Visualizations Used

- **Histograms**: Distribution and outlier detection
- **Box Plots**: Feature spread across obesity types
- **Count Plots**: Categorical features vs obesity levels
- **Heatmap**: Pearson Correlation Matrix
- **PCA (2D)**: Dimensionality reduction for class separation

### 🔍 Key Observations

- **Strong Correlations**:
  - `Weight` & `NObeyesdad`: `0.91`
  - `BMI` & `NObeyesdad`: `0.72`
- **Lifestyle Insights**:
  - 🚶 Low activity, 🛻 sedentary transport, 🍔 fast food → high obesity
  - 🥦 Vegetable & 💧 water intake → linked to healthier weight

---

## 🤖 PART THREE: Modeling & Insights

### 🔧 Models Evaluated

| Model               | Accuracy | Precision | F1 Score |
|--------------------|----------|-----------|----------|
| KNN (n=3)          | 0.8578   | 0.8638    | 0.8553   |
| KNN (n=5, dist.)   | 0.8531   | 0.8612    | 0.8491   |
| Random Forest (100)| **0.9668** | **0.9710** | **0.9673** |
| Random Forest (200, depth=10) | 0.9526 | 0.9557 | 0.9527 |

### ✅ Best Model: Random Forest (n_estimators=100)

- Excellent accuracy (~97%)
- Handles non-linear, high-dimensional data
- Robust against noise and overfitting

### ⚖️ Model Comparison

| Model             | Pros                                | Cons                                 |
|------------------|-------------------------------------|--------------------------------------|
| **KNN**          | Simple, no training needed          | Sensitive to outliers, slow for large datasets |
| **Logistic Reg.**| Fast, interpretable                 | Struggles with non-linearity         |
| **SVM**          | Good with high-dimensional data     | Computationally expensive            |
| **Random Forest**| High accuracy, robust to noise      | Less interpretable, higher complexity |

---

## 🚀 Future Improvements

- 🔍 **GridSearchCV** for automated hyperparameter tuning
- 🛠️ **Enhanced Feature Engineering**: interaction terms, domain-specific features
- 🧪 **Test More Models**: Ensemble methods, Neural Networks
- 🧑‍💻 **GUI/Web Integration**: Deploy as desktop or web health tool

---

## 📌 Conclusion

This project demonstrates how careful preprocessing, exploratory analysis, and machine learning can effectively predict obesity levels and uncover lifestyle patterns. The pipeline is structured, extendable, and ready for integration in real-world health monitoring systems.

---

## 📚 References

- [Obesity Dataset on Kaggle / UCI]
- Scikit-learn, Pandas, Seaborn, Matplotlib, NumPy
- WHO, CDC studies on lifestyle and obesity
