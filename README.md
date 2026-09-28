# House Price Prediction with Linear Regression

Predicting house sale prices from **square footage, bedrooms and bathrooms** using multiple linear regression, with a focus on checking whether the model can be trusted, not only fitting it.

## The Task

> Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

This project keeps to those three inputs. The extra effort goes into building them properly, validating the model, and interpreting the results.

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle (residential homes in Ames, Iowa).

| Input | Source column | Notes |
|---|---|---|
| Square footage | `GrLivArea` | Above-ground living area, in sq ft |
| Bedrooms | `BedroomAbvGr` | Bedrooms above basement level |
| Bathrooms | `TotalBaths` (engineered) | `FullBath + 0.5 x HalfBath + BsmtFullBath + 0.5 x BsmtHalfBath` |
| **Target** | `SalePrice` | Sale price in dollars |

The dataset stores bathrooms in four separate columns. Counting only `FullBath` would ignore half baths and basement baths, so they are combined into one feature, with half baths counted as 0.5.

## Approach

1. **Explore** each input against price (scatter and box plots)
2. **Check input overlap** with a correlation heatmap and VIF, since bigger houses tend to have more bedrooms and baths
3. **Handle outliers**: a few very large houses sold unusually cheaply and pull the fitted line toward them
4. **Split** the cleaned data into training and test sets
5. **Baseline**: a model that always predicts the average price, which any real model has to beat
6. **Train** a `Pipeline` (StandardScaler + LinearRegression)
7. **Validate** with the test set and 5-fold cross-validation
8. **Interpret**: coefficients in dollars, standardized coefficients to rank the inputs, and p-values with confidence intervals (statsmodels)
9. **Compare variants**: outliers removed vs kept, and predicting `log(price)`
10. **Diagnose** with predicted-vs-actual and residual plots
11. **Use it**: a `predict_price(sqft, bedrooms, bathrooms)` function, and predictions on Kaggle's `test.csv`

## How to Run

1. Download `train.csv` and `test.csv` from the Kaggle competition page.
2. Open `House_Price_Prediction_Linear_Regression.ipynb` in [Google Colab](https://colab.research.google.com).
3. Upload both CSV files using the folder icon in the left sidebar.
4. Choose **Runtime > Run all**.

The notebook expects the files at `/content/train.csv` and `/content/test.csv`. If yours are elsewhere, change `TRAIN_PATH` and `TEST_PATH` in Section 2.

**Requirements:** Python 3, pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn (all preinstalled in Colab).

## Results

All comparisons use the same held-out test set of 292 houses (outliers removed).

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Baseline (average price) | 61,963 | 87,740 | -0.008 |
| Linear regression (outliers removed) | 36,618 | 49,921 | 0.674 |
| Linear regression (outliers kept) | 36,149 | 50,559 | 0.665 |
| **Linear regression on log(price)** | **32,660** | **47,288** | **0.707** |

**5-fold cross-validation (linear model):** mean R² = 0.642 (+/- 0.04, folds ranged 0.573 to 0.684), mean RMSE = $45,597

**Coefficients (statsmodels OLS on the training set, R² = 0.655):**

| Input | Coefficient | 95% confidence interval | p-value |
|---|---|---|---|
| Square footage (per sq ft) | +$114.55 | $107 to $122 | < 0.001 |
| Bedrooms (per bedroom) | -$29,070 | -$33,000 to -$25,200 | < 0.001 |
| Bathrooms (per bathroom) | +$26,310 | $22,200 to $30,500 | < 0.001 |

### Key findings

- **The model clearly beats the baseline.** Test RMSE falls from about $87,700 to $49,900, a reduction of roughly 43%.
- **Predicting log(price) was the best variant.** It improved every metric on this test split: R² from 0.674 to 0.707, MAE by about $4,000, RMSE by about $2,600. The residuals are right-skewed (skew 0.75, kurtosis 7.5), which is the situation a log transform is meant to help with.
- **Removing the two outlier houses made little difference.** RMSE and R² improved slightly, while MAE got slightly worse, so the evidence is mixed rather than a clear win. Two houses out of about 1,460 have a small effect on these metrics.
- **The bedroom coefficient is negative and statistically significant (about -$29,000 per bedroom).** This does not mean bedrooms lower a home's value. It is the effect of one more bedroom with square footage and bathrooms held fixed: the same floor area split into more, smaller rooms. Under this model, an extra bedroom only pays for itself if it comes with roughly 250 more square feet (29,070 / 114.55).
- **Cross-validation is the more cautious estimate.** The mean CV R² (0.642) is a little below the single-split test R² (0.674), and the test score sits within one standard deviation of the CV mean.

## Limitations

- **Three inputs leave most of the story out.** Location, build quality, condition and age all affect price and are not in the model, so a large share of the variation is unexplained.
- **Linearity is an assumption.** A straight-line relationship between size and price will not hold equally across cheap and expensive houses. The residual plot shows where it strains.
- **The residuals are not normally distributed** (Jarque-Bera test strongly rejects normality, skew 0.75, kurtosis 7.5), so the p-values and confidence intervals are approximate. The very small p-values are unlikely to change, but the exact intervals should not be over-read.
- **The outlier rule was chosen by inspecting the data** (`GrLivArea > 4000` and price below $300,000). It is a judgment call, and the notebook shows how much the model changes with and without it.
- **Local data.** The houses are from one city over a few years, so predictions should not be applied to other markets.
- **Predictions outside the training range are unreliable.** `predict_price()` prints a warning when an input falls outside what the model saw.


## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn
