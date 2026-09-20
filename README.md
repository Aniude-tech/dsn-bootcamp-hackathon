# DSN Bootcamp ML Track - Sales Prediction

## Problem Statement
Predict total product sales at DSN Mart store locations based on product and outlet characteristics. This model helps optimize inventory, pricing, and store investment decisions across Nigeria's retail network.

## Dataset Overview
- **Training Data:** 6,818 records × 13 features
- **Target Variable:** `total_sales` (complete, no missing values)

## Data Quality
| Feature | Missing (%) | Issue |
|---------|------------|-------|
| product_weight_kg | 17.97% | Will impute with median |
| store_size | 28.15% | Will impute with mode |
| All Others | 0% | Clean |

## Feature Breakdown

**Numerical Features (5):**
- product_weight_kg, shelf_visibility, product_price, store_age_years, total_sales

**Categorical Features (8):**
- Fat Content: 2 categories (Low Fat, Regular)
- Product Category: 48 categories (inconsistent casing — needs standardization)
- Store Size: 3 categories (Large, Small, Medium) + missing values
- Store Location Tier: 3 tiers (Tier 1, Tier 2, Tier 3)
- Store Format: 4 types (Standard Supermarket: 4,462 | Corner Shop: 866 | Flagship Hypermarket: 748 | Superstore: 742)

## Key Insights from EDA

**Strongest Driver of Sales:**
- `product_price` shows strongest correlation with total_sales (r = 0.57)
- Other numerical features have weak correlation

**Data Patterns:**
- Standard Supermarket dominates (65% of stores)
- Tier 3 locations most common (39% of stores)
- Product categories need standardization (mixed case, spelling variations)

## Approach

1. **Data Cleaning** → Handle missing values, standardize product categories

## Data Cleaning

**Missing Values Handled:**
- product_weight_kg (17.97%) → Imputed with median
- store_size (28.15%) → Imputed with mode (most frequent)

**Data Quality Checks:**
- No duplicate rows found
- Product categories standardized to lowercase
- Shelf visibility outlier (0.32) retained — represents high-visibility product placement
**Result:** Clean dataset (6,818 rows, 0 missing values) ready for feature engineering

2. **Feature Engineering** → Encode categories, create interaction features

**Encoding Strategy:**
- Numerical features: StandardScaler normalization
- Categorical features: OneHotEncoder (4 categories × 5 features = expanded feature set)

**Multicollinearity Analysis:**
High VIF detected in 8 categorical features (store_size, store_location_tier, store_format). This is expected with one-hot encoding — categorical variables are inherently interdependent.

**Mitigation Strategy:**
- Tree-based models (Random Forest, XGBoost): **Not affected** — inherently robust to multicollinearity
- Linear Regression: Uses **Ridge regularization** to handle multicollinearity
- **No features dropped** — all information retained
**Result:** 28 total features originally 9 features before one hot encoding

3. **Model Selection** → Test Linear Regression, Random Forest, XGBoost

## Model Building & Evaluation

**Models Tested:**
1. Ridge Regression: 1123.44 RMSE (R² = 0.571)
2. Random Forest: 1123.21 RMSE (R² = 0.572) — Overfitting detected
3. XGBoost: 1107.12 RMSE (R² = 0.584) — **Best performer**

**Key Findings:**
- XGBoost provides best validation performance with lowest RMSE
- Random Forest shows overfitting (Train RMSE: 587.4 vs Val RMSE: 1123.2)
- Ridge Regression inadequate — linear relationships insufficient for sales prediction
- Product price identified as strongest predictor of sales

**Best Model Selected:** XGBoost Regressor
4. **Hyperparameter Tuning** → Optimize best performer
5. **Evaluation** → Minimize RMSE on test set

## Model Performance
(To be updated after training)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
# Run notebooks in order: 01 → 02 → 03 → 04 → 05
```

## Author
Paul | Data scientist | September 2026