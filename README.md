# Q10_predict

R code for the machine-learning analysis in:

> Xiao, L. et al. *Mineral-associated blue carbon is unexpectedly vulnerable to warming.*

## Contents

`Q10_Complete_Analysis_3.R` — complete pipeline: model training, performance comparison, and
interpretability analysis (feature importance and partial dependence) of the top-performing models.

## Input

`rPCA_scores.csv` — one row per site (n = 116), containing:

- `Q10`: response variable
- `rPC1`–`rPC7`: predictors (rotated principal component scores summarising 22 environmental
  variables selected by VIF < 5; see Methods, "Statistics")

## What the script does

1. Splits the data into training (70%) and test (30%) sets by stratified random sampling
   (`createDataPartition`, `set.seed(456)`).
2. Trains 13 regression models with 10-fold cross-validation within the training set
   (5-fold for XGBoost): linear regression, GLM, random forest, SVM, GBM, MARS, neural network,
   PLSR, XGBoost, LightGBM, BART, EBM, and a stacking ensemble.
   *Note: the paper reports the 11-model comparison; XGBoost and the stacking ensemble were
   exploratory and are not included in the manuscript.*
3. Evaluates all models on the held-out test set (R², RMSE, MAE) and plots performance.
4. For the top-two models: relative feature importance, partial dependence plots, and
   observed-vs-predicted scatter plots.

## Requirements

- R (>= 4.3) with packages: `caret`, `randomForest`, `e1071`, `kernlab`, `gbm`, `earth`, `nnet`,
  `pls`, `xgboost`, `lightgbm`, `BART`, `caretEnsemble`, `ggplot2`, `Metrics`, `reticulate`,
  `pdp`, `parallel`
- Python with the `interpret` package (for EBM), accessed via `reticulate`
  (a conda env named `r-reticulate` is used by default)

Edit the working directory (`setwd`) and, if needed, the Python path at the top of the script
before running.

## Usage

```r
source("Q10_Complete_Analysis_3.R")
```

Run time is minutes on a standard desktop; BART and partial-dependence calculations use
optional multi-core parallelisation on non-Windows systems.

## Outputs

- `intermediate_model_*.rds` — fitted model objects and test-set predictions for every model
- `Model_Performance_Points_with_Lines.pdf` — cross-validated and test-set performance of all models
- `Observed_vs_Predicted_All_Models.pdf` — observed-vs-predicted scatter for all models
- `Figure_1*_Feature_Importance.pdf`, `Figure_1*_PDP_Paper_Style.pdf`, `Figure_1*_All_Data_Scatter.pdf`
  — interpretability figures for the top-two models


