# ============================================================================
# Q10数据完整分析流程
# 从13模型训练到前两位模型可解释性分析（PDF版本）
# ============================================================================
# 包含：
# 1. 13个模型训练
# 2. 模型性能对比可视化
# 3. 模型性能汇总与可视化
# 4. 前两位模型特征重要性、PDP及全数据散点图
# 5. 前两位模型可解释性分析（特征重要性用绝对值）
# 所有图表保存为PDF格式，所有中间数据保存
# ============================================================================

setwd("D:/Claude/Q10/Q10_model_116_v2")

cat("============================================================\n")
cat("Q10完整分析流程启动\n")
cat("13模型训练 + 前两位模型可解释性分析\n")
cat("============================================================\n\n")

# ============================================================================
# 第一部分：包管理与环境配置
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第一部分：加载R包和配置Python环境\n")
cat("=" , rep("=", 68), "\n\n", sep="")

# 加载必要的包
suppressPackageStartupMessages({
  library(caret)
  library(randomForest)
  library(e1071)
  library(kernlab)
  library(gbm)
  library(earth)
  library(nnet)
  library(pls)
  library(xgboost)
  library(lightgbm)
  library(BART)
  library(caretEnsemble)
  library(ggplot2)
  library(Metrics)
  library(reticulate)
  library(parallel)
})

cat("✓ 所有R包加载完成\n\n")

# 定义R²计算函数
calc_R2 <- function(pred, obs) {
  ss_res <- sum((obs - pred)^2)
  ss_tot <- sum((obs - mean(obs))^2)
  return(1 - ss_res / ss_tot)
}

# 配置Python环境
cat("配置Python环境...\n")

python_path <- "C:/Users/Lenovo/AppData/Local/r-miniconda/envs/r-reticulate/python.exe"

if (file.exists(python_path)) {
  use_python(python_path, required = TRUE)
  cat("✓ Python环境已配置\n")
  cat("  路径:", py_config()$python, "\n\n")
} else {
  use_condaenv("r-reticulate", required = TRUE)
  cat("✓ 使用conda环境\n\n")
}

# 导入Python包
interpret_glassbox <- import("interpret.glassbox")
ebm_available <- TRUE

cat("✓ interpret包已加载\n\n")

# ============================================================================
# 第二部分：数据加载与划分
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第二部分：数据加载与训练/测试划分\n")
cat("=" , rep("=", 68), "\n\n", sep="")

# 加载数据
data <- read.csv("rPCA_scores.csv", header = TRUE)

cat("数据维度:", nrow(data), "行", ncol(data), "列\n")

# 提取特征和目标变量
X <- data[, grep("^rPC", colnames(data))]
y <- data$Q10
feature_names <- colnames(X)

cat("特征数量:", ncol(X), "\n")
cat("特征名称:", paste(feature_names, collapse = ", "), "\n")
cat("样本数量:", nrow(X), "\n")
cat("目标变量范围: [", round(min(y), 2), ",", round(max(y), 2), "]\n\n")

# 数据划分（70/30）
set.seed(456)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)

X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

train_data <- data.frame(Q10 = y_train, X_train)
test_data <- data.frame(Q10 = y_test, X_test)

cat("训练集样本数:", length(y_train), "\n")
cat("测试集样本数:", length(y_test), "\n\n")

# 交叉验证设置
cv_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  returnResamp = "final"
)

cv_control_5fold <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  returnResamp = "final"
)

cat("交叉验证设置:\n")
cat("  标准CV: 10折\n")
cat("  XGBoost专用CV: 5折\n\n")

# ============================================================================
# 第三部分：13个模型训练
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第三部分：训练13个机器学习模型\n")
cat("=" , rep("=", 68), "\n\n", sep="")

models <- list()
predictions <- list()
performance <- data.frame(
  Model = character(),
  R2 = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  stringsAsFactors = FALSE
)

model_count <- 0

# 模型1: 线性回归
cat("[1/13] 线性回归 (LR)...\n")
set.seed(123)
models$LR <- train(Q10 ~ ., data = train_data, method = "lm", trControl = cv_control)
predictions$LR <- predict(models$LR, test_data)
performance <- rbind(performance, data.frame(
  Model = "Linear Regression",
  R2 = calc_R2(predictions$LR, y_test),
  RMSE = Metrics::rmse(predictions$LR, y_test),
  MAE = Metrics::mae(predictions$LR, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$LR, predictions = predictions$LR,
             performance = performance[model_count,]),
        "intermediate_model_01_LR.rds")
cat("  ✓ R² =", round(calc_R2(predictions$LR, y_test), 4), "\n\n")

# 模型2: GLM
cat("[2/13] 广义线性模型 (GLM)...\n")
set.seed(123)
models$GLM <- train(Q10 ~ ., data = train_data, method = "glm", trControl = cv_control)
predictions$GLM <- predict(models$GLM, test_data)
performance <- rbind(performance, data.frame(
  Model = "GLM",
  R2 = calc_R2(predictions$GLM, y_test),
  RMSE = Metrics::rmse(predictions$GLM, y_test),
  MAE = Metrics::mae(predictions$GLM, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$GLM, predictions = predictions$GLM,
             performance = performance[model_count,]),
        "intermediate_model_02_GLM.rds")
cat("  ✓ R² =", round(calc_R2(predictions$GLM, y_test), 4), "\n\n")

# 模型3: 随机森林
cat("[3/13] 随机森林 (RF)...\n")
set.seed(123)
models$RF <- train(
  Q10 ~ ., data = train_data, method = "rf", trControl = cv_control,
  tuneGrid = expand.grid(mtry = c(2, 3, 4, 5)), ntree = 500
)
predictions$RF <- predict(models$RF, test_data)
performance <- rbind(performance, data.frame(
  Model = "Random Forest",
  R2 = calc_R2(predictions$RF, y_test),
  RMSE = Metrics::rmse(predictions$RF, y_test),
  MAE = Metrics::mae(predictions$RF, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$RF, predictions = predictions$RF,
             performance = performance[model_count,]),
        "intermediate_model_03_RF.rds")
cat("  ✓ R² =", round(calc_R2(predictions$RF, y_test), 4), "\n\n")

# 模型4: SVM
cat("[4/13] 支持向量机 (SVM)...\n")
set.seed(123)
models$SVM <- train(
  Q10 ~ ., data = train_data, method = "svmRadial", trControl = cv_control,
  tuneGrid = expand.grid(sigma = c(0.01, 0.1, 1), C = c(0.1, 1, 10))
)
predictions$SVM <- predict(models$SVM, test_data)
performance <- rbind(performance, data.frame(
  Model = "SVM",
  R2 = calc_R2(predictions$SVM, y_test),
  RMSE = Metrics::rmse(predictions$SVM, y_test),
  MAE = Metrics::mae(predictions$SVM, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$SVM, predictions = predictions$SVM,
             performance = performance[model_count,]),
        "intermediate_model_04_SVM.rds")
cat("  ✓ R² =", round(calc_R2(predictions$SVM, y_test), 4), "\n\n")

# 模型5: GBM
cat("[5/13] 梯度提升机 (GBM)...\n")
set.seed(123)
models$GBM <- train(
  Q10 ~ ., data = train_data, method = "gbm", trControl = cv_control,
  tuneGrid = expand.grid(
    n.trees = c(100, 200),
    interaction.depth = c(1, 3),
    shrinkage = 0.1,
    n.minobsinnode = 10
  ),
  verbose = FALSE
)
predictions$GBM <- predict(models$GBM, test_data)
performance <- rbind(performance, data.frame(
  Model = "GBM",
  R2 = calc_R2(predictions$GBM, y_test),
  RMSE = Metrics::rmse(predictions$GBM, y_test),
  MAE = Metrics::mae(predictions$GBM, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$GBM, predictions = predictions$GBM,
             performance = performance[model_count,]),
        "intermediate_model_05_GBM.rds")
cat("  ✓ R² =", round(calc_R2(predictions$GBM, y_test), 4), "\n\n")

# 模型6: MARS
cat("[6/13] 多元自适应回归样条 (MARS)...\n")
set.seed(123)
models$MARS <- train(
  Q10 ~ ., data = train_data, method = "earth", trControl = cv_control,
  tuneGrid = expand.grid(degree = c(1, 2), nprune = c(5, 10, 15))
)
predictions$MARS <- predict(models$MARS, test_data)
performance <- rbind(performance, data.frame(
  Model = "MARS",
  R2 = calc_R2(predictions$MARS, y_test),
  RMSE = Metrics::rmse(predictions$MARS, y_test),
  MAE = Metrics::mae(predictions$MARS, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$MARS, predictions = predictions$MARS,
             performance = performance[model_count,]),
        "intermediate_model_06_MARS.rds")
cat("  ✓ R² =", round(calc_R2(predictions$MARS, y_test), 4), "\n\n")

# 模型7: 神经网络
cat("[7/13] 神经网络 (NNET)...\n")
set.seed(123)
models$NNET <- train(
  Q10 ~ ., data = train_data, method = "nnet", trControl = cv_control,
  tuneGrid = expand.grid(size = c(3, 5, 7), decay = c(0.01, 0.1)),
  linout = TRUE, trace = FALSE, maxit = 500
)
predictions$NNET <- predict(models$NNET, test_data)
performance <- rbind(performance, data.frame(
  Model = "Neural Network",
  R2 = calc_R2(predictions$NNET, y_test),
  RMSE = Metrics::rmse(predictions$NNET, y_test),
  MAE = Metrics::mae(predictions$NNET, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$NNET, predictions = predictions$NNET,
             performance = performance[model_count,]),
        "intermediate_model_07_NNET.rds")
cat("  ✓ R² =", round(calc_R2(predictions$NNET, y_test), 4), "\n\n")

# 模型8: PLSR
cat("[8/13] 偏最小二乘回归 (PLSR)...\n")
set.seed(123)
models$PLSR <- train(
  Q10 ~ ., data = train_data, method = "pls", trControl = cv_control,
  tuneGrid = expand.grid(ncomp = 1:min(5, ncol(X_train)))
)
predictions$PLSR <- predict(models$PLSR, test_data)
performance <- rbind(performance, data.frame(
  Model = "PLSR",
  R2 = calc_R2(predictions$PLSR, y_test),
  RMSE = Metrics::rmse(predictions$PLSR, y_test),
  MAE = Metrics::mae(predictions$PLSR, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$PLSR, predictions = predictions$PLSR,
             performance = performance[model_count,]),
        "intermediate_model_08_PLSR.rds")
cat("  ✓ R² =", round(calc_R2(predictions$PLSR, y_test), 4), "\n\n")

# 模型9: XGBoost（5折CV）
cat("[9/13] XGBoost（使用5折交叉验证）...\n")
xgboost_success <- FALSE
tryCatch({
  set.seed(123)
  models$XGBoost <- train(
    Q10 ~ ., data = train_data, method = "xgbTree",
    trControl = cv_control_5fold,
    tuneGrid = expand.grid(
      nrounds = 50,
      max_depth = 2,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 5,
      subsample = 0.8
    ),
    verbose = FALSE
  )
  predictions$XGBoost <- predict(models$XGBoost, test_data)
  performance <- rbind(performance, data.frame(
    Model = "XGBoost",
    R2 = calc_R2(predictions$XGBoost, y_test),
    RMSE = Metrics::rmse(predictions$XGBoost, y_test),
    MAE = Metrics::mae(predictions$XGBoost, y_test)
  ))
  model_count <- model_count + 1
  saveRDS(list(model = models$XGBoost, predictions = predictions$XGBoost,
               performance = performance[model_count,]),
          "intermediate_model_09_XGBoost.rds")
  cat("  ✓ R² =", round(calc_R2(predictions$XGBoost, y_test), 4), "\n\n")
  xgboost_success <- TRUE
}, error = function(e) {
  cat("  ✗ XGBoost训练失败（样本量不足）\n\n")
})

# 模型10: LightGBM
cat("[10/13] LightGBM...\n")
set.seed(123)
lgb_train <- lgb.Dataset(
  data = as.matrix(X_train),
  label = y_train
)
params <- list(
  objective = "regression",
  metric = "rmse",
  num_leaves = 15,
  learning_rate = 0.05,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  verbose = -1
)
models$LightGBM <- lgb.train(
  params = params,
  data = lgb_train,
  nrounds = 100,
  valids = list(train = lgb_train),
  early_stopping_rounds = 20,
  verbose = -1
)
predictions$LightGBM <- predict(models$LightGBM, as.matrix(X_test))
performance <- rbind(performance, data.frame(
  Model = "LightGBM",
  R2 = calc_R2(predictions$LightGBM, y_test),
  RMSE = Metrics::rmse(predictions$LightGBM, y_test),
  MAE = Metrics::mae(predictions$LightGBM, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$LightGBM, predictions = predictions$LightGBM,
             performance = performance[model_count,]),
        "intermediate_model_10_LightGBM.rds")
cat("  ✓ R² =", round(calc_R2(predictions$LightGBM, y_test), 4), "\n\n")

# 模型11: BART（优化版：加速训练和预测）
cat("[11/13] 贝叶斯加法回归树 (BART)...\n")

# 检测可用核心数和操作系统
n_cores <- parallel::detectCores()
use_cores <- max(1, min(n_cores - 1, 4))  # 使用最多4个核心，保留1个核心

set.seed(123)

# Windows系统不支持mc.wbart，使用wbart；Unix系统使用mc.wbart多核加速
if (.Platform$OS.type == "windows") {
  cat(sprintf("  Windows系统，使用单核wbart（检测到%d个CPU核心）\n", n_cores))

  bart_model <- wbart(
    x.train = as.matrix(X_train),
    y.train = y_train,
    x.test = as.matrix(X_test),
    ndpost = 200,      # 从1000减少到200（预测提速5倍）
    nskip = 100,       # 从200减少到100
    ntree = 100,       # 从默认200减少到100（再提速2倍）
    printevery = 1000
  )

} else {
  # Unix/Linux/Mac系统可以使用多核
  cat(sprintf("  检测到%d个CPU核心，使用%d个核心并行计算\n", n_cores, use_cores))

  bart_model <- mc.wbart(
    x.train = as.matrix(X_train),
    y.train = y_train,
    x.test = as.matrix(X_test),
    ndpost = 200,      # 从1000减少到200（预测提速5倍）
    nskip = 100,       # 从200减少到100
    ntree = 100,       # 从默认200减少到100（再提速2倍）
    mc.cores = use_cores,  # 多核并行
    printevery = 1000
  )
}

models$BART <- bart_model
predictions$BART <- bart_model$yhat.test.mean
performance <- rbind(performance, data.frame(
  Model = "BART",
  R2 = calc_R2(predictions$BART, y_test),
  RMSE = Metrics::rmse(predictions$BART, y_test),
  MAE = Metrics::mae(predictions$BART, y_test)
))
model_count <- model_count + 1
saveRDS(list(model = models$BART, predictions = predictions$BART,
             performance = performance[model_count,]),
        "intermediate_model_11_BART.rds")
cat("  ✓ R² =", round(calc_R2(predictions$BART, y_test), 4), "\n")
cat("  参数优化: ndpost=200, ntree=100\n")
cat("  预测速度预计提升约10倍\n\n")

# 模型12: EBM
cat("[12/13] 可解释提升机 (EBM)...\n")
set.seed(123)
ebm_model <- interpret_glassbox$ExplainableBoostingRegressor(
  interactions = 0L,
  max_bins = 256L,
  max_rounds = 5000L,
  learning_rate = 0.01,
  min_samples_leaf = 2L,
  random_state = 123L
)
ebm_model$fit(as.matrix(X_train), y_train)
predictions$EBM <- ebm_model$predict(as.matrix(X_test))
models$EBM <- ebm_model
performance <- rbind(performance, data.frame(
  Model = "EBM",
  R2 = calc_R2(predictions$EBM, y_test),
  RMSE = Metrics::rmse(predictions$EBM, y_test),
  MAE = Metrics::mae(predictions$EBM, y_test)
))
model_count <- model_count + 1

# 保存EBM模型和全局解释对象
ebm_global <- ebm_model$explain_global()
saveRDS(list(
  model = ebm_model,
  predictions = predictions$EBM,
  performance = performance[model_count,],
  global_explanation = ebm_global
), "intermediate_model_12_EBM.rds")

cat("  ✓ R² =", round(calc_R2(predictions$EBM, y_test), 4), "\n\n")

# 模型13: Stacking
cat("[13/13] Stacking集成模型...\n")
tryCatch({
  model_list <- caretList(
    Q10 ~ ., data = train_data,
    trControl = cv_control,
    methodList = c("lm", "glm"),
    tuneList = list(
      rf = caretModelSpec(method = "rf", tuneGrid = data.frame(mtry = 3))
    )
  )
  stacking_model <- caretStack(
    model_list,
    method = "lm",
    metric = "RMSE",
    trControl = cv_control
  )
  models$Stacking <- stacking_model
  predictions$Stacking <- as.vector(predict(stacking_model, test_data))
  performance <- rbind(performance, data.frame(
    Model = "Stacking",
    R2 = calc_R2(predictions$Stacking, y_test),
    RMSE = Metrics::rmse(predictions$Stacking, y_test),
    MAE = Metrics::mae(predictions$Stacking, y_test)
  ))
  model_count <- model_count + 1
  saveRDS(list(model = models$Stacking, predictions = predictions$Stacking,
               performance = performance[model_count,]),
          "intermediate_model_13_Stacking.rds")
  cat("  ✓ R² =", round(calc_R2(predictions$Stacking, y_test), 4), "\n\n")
}, error = function(e) {
  cat("  ✗ Stacking训练失败\n\n")
})

# 排序性能表
performance <- performance[order(-performance$R2), ]
rownames(performance) <- NULL

cat("成功训练的模型数量:", model_count, "/13\n\n")

# ============================================================================
# 第四部分：模型性能汇总与可视化
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第四部分：模型性能汇总与可视化（PDF格式）\n")
cat("=" , rep("=", 68), "\n\n", sep="")

# 保存性能表
write.csv(performance, "Model_Performance_Final.csv", row.names = FALSE)
cat("✓ Model_Performance_Final.csv\n\n")

# 保存预测值
predictions_df <- data.frame(
  ID = data$ID[-train_index],
  Observed = y_test
)

# 确保所有预测值都是向量格式
for (model_name in names(predictions)) {
  pred_values <- predictions[[model_name]]

  tryCatch({
    # 处理Python对象（如EBM预测，numpy数组等）
    if (inherits(pred_values, "python.builtin.object") ||
        inherits(pred_values, "numpy.ndarray")) {
      pred_values <- as.numeric(pred_values)
    }

    # 处理list类型
    if (is.list(pred_values) && !is.data.frame(pred_values)) {
      pred_values <- unlist(pred_values)
    }

    # 处理matrix类型
    if (is.matrix(pred_values)) {
      pred_values <- as.vector(pred_values)
    }

    # 最终确保是数值向量
    pred_values <- as.numeric(as.vector(pred_values))

    # 检查长度是否匹配
    if (length(pred_values) != nrow(predictions_df)) {
      warning(sprintf("Model %s: prediction length mismatch. Expected %d, got %d",
                     model_name, nrow(predictions_df), length(pred_values)))
    }

    predictions_df[[model_name]] <- pred_values

  }, error = function(e) {
    warning(sprintf("Failed to add predictions for model %s: %s",
                   model_name, e$message))
    # 添加NA列作为占位符
    predictions_df[[model_name]] <- rep(NA, nrow(predictions_df))
  })
}

write.csv(predictions_df, "Model_Predictions_Final.csv", row.names = FALSE)
cat("✓ Model_Predictions_Final.csv\n\n")

cat("模型性能排名（Top 5）：\n")
print(head(performance, 5), row.names = FALSE)
cat("\n")

cat("生成性能对比圆圈竖线图（PDF）...\n")
pdf("Model_Performance_Points_with_Lines.pdf", width = 16, height = 8)
par(mfrow = c(1, 3), mar = c(12, 4, 3, 2))
# 定义颜色
cols <- rainbow(nrow(performance))
# ---- R² ----
plot(
  x = 1:nrow(performance),
  y = performance$R2,
  ylim = c(min(0, min(performance$R2)), max(performance$R2) * 1.1),
  xaxt = "n",
  type = "n",  # 不画点，先建立空图
  main = paste0("R² Comparison (", nrow(performance), " Models)"),
  ylab = "R²",
  xlab = ""
)
abline(h = 0, col = "gray", lty = 2)
# 添加竖线
segments(
  x0 = 1:nrow(performance),
  y0 = 0,
  x1 = 1:nrow(performance),
  y1 = performance$R2,
  col = "gray",
  lwd = 3
)
# 添加圆圈（点）
points(
  x = 1:nrow(performance),
  y = performance$R2,
  pch = 21,          # 有填充和边框的圆
  bg = cols,
  col = "black",
  cex = 3            # 圆圈更大
)
# 轴与数值
axis(1, at = 1:nrow(performance), labels = performance$Model, las = 2, cex.axis = 0.7)
text(
  x = 1:nrow(performance),
  y = performance$R2 + ifelse(performance$R2 > 0, max(performance$R2) * 0.03, -max(performance$R2) * 0.03),
  labels = round(performance$R2, 3),
  cex = 0.8
)
# ---- RMSE ----
plot(
  x = 1:nrow(performance),
  y = performance$RMSE,
  ylim = c(0, max(performance$RMSE) * 1.1),
  xaxt = "n",
  type = "n",
  main = "RMSE Comparison",
  ylab = "RMSE",
  xlab = ""
)
# 竖线
segments(
  x0 = 1:nrow(performance),
  y0 = 0,
  x1 = 1:nrow(performance),
  y1 = performance$RMSE,
  col = "gray",,
  lwd = 3
)
# 圆点
points(
  x = 1:nrow(performance),
  y = performance$RMSE,
  pch = 21,
  bg = cols,
  col = "black",
  cex = 3
)
axis(1, at = 1:nrow(performance), labels = performance$Model, las = 2, cex.axis = 0.7)
text(
  x = 1:nrow(performance),
  y = performance$RMSE + max(performance$RMSE) * 0.03,
  labels = round(performance$RMSE, 3),
  cex = 0.8
)
# ---- MAE ----
plot(
  x = 1:nrow(performance),
  y = performance$MAE,
  ylim = c(0, max(performance$MAE) * 1.1),
  xaxt = "n",
  type = "n",
  main = "MAE Comparison",
  ylab = "MAE",
  xlab = ""
)
segments(
  x0 = 1:nrow(performance),
  y0 = 0,
  x1 = 1:nrow(performance),
  y1 = performance$MAE,
  col = "gray",
  lwd = 3
)
points(
  x = 1:nrow(performance),
  y = performance$MAE,
  pch = 21,
  bg = cols,
  col = "black",
  cex = 3
)
axis(1, at = 1:nrow(performance), labels = performance$Model, las = 2, cex.axis = 0.7)
text(
  x = 1:nrow(performance),
  y = performance$MAE + max(performance$MAE) * 0.03,
  labels = round(performance$MAE, 3),
  cex = 0.8
)
dev.off()
cat("✓ Model_Performance_Points_with_Lines.pdf\n\n")

# 观测值vs预测值散点图（PDF）
cat("生成观测值vs预测值散点图（PDF）...\n")

valid_models <- names(predictions)
n_valid <- length(valid_models)
n_cols <- 4
n_rows <- ceiling(n_valid / n_cols)

pdf("Observed_vs_Predicted_All_Models.pdf",
    width = 4 * n_cols, height = 4 * n_rows)
par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 3, 1))

for (model_name in valid_models) {
  # 找到对应的性能信息
  perf_row <- performance[performance$Model == model_name |
                          grepl(model_name, performance$Model, ignore.case = TRUE), ]

  # 提取预测值并确保是数值向量
  pred_values <- predictions[[model_name]]

  # 处理潜在的list或非数值问题
  tryCatch({
    if (is.list(pred_values) && !is.data.frame(pred_values)) {
      pred_values <- unlist(pred_values)
    }
    pred_values <- as.numeric(pred_values)

    # 获取R²和显示名称
    if (nrow(perf_row) > 0) {
      r2_value <- perf_row$R2[1]
      display_name <- perf_row$Model[1]
    } else {
      r2_value <- calc_R2(pred_values, y_test)
      display_name <- model_name
    }

    plot(
      y_test,
      pred_values,
      main = paste0(display_name, "\nR² = ", round(r2_value, 3)),
      xlab = "Observed Q10",
      ylab = "Predicted Q10",
      pch = 19,
      col = rgb(0, 0, 1, 0.5),
      cex = 1.2
    )

    abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
    abline(lm(pred_values ~ y_test), col = "blue", lwd = 2)
    grid()

  }, error = function(e) {
    plot.new()
    text(0.5, 0.5, paste("Error:", model_name))
  })
}

dev.off()
cat("✓ Observed_vs_Predicted_All_Models.pdf\n\n")

# ============================================================================
# 第五部分：前两位模型可解释性分析
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第五部分：前两位模型可解释性分析（PDF格式）\n")
cat("=" , rep("=", 68), "\n\n", sep="")

for (rank_i in 1:min(2, nrow(performance))) {

  cat(sprintf("\n=== [%d/2] 分析第%d名模型 ===\n", rank_i, rank_i))

  best_model_name <- performance$Model[rank_i]
  model_rank_prefix <- sprintf("Rank%d_%s", rank_i, gsub(" ", "_", best_model_name))

  cat("模型名称:", best_model_name, "\n")
  cat(sprintf("R² = %.4f, RMSE = %.4f, MAE = %.4f\n\n",
              performance$R2[rank_i], performance$RMSE[rank_i], performance$MAE[rank_i]))

  # 根据模型名称找到对应的模型对象键值（显式映射）
  perf_to_key <- c(
    "Linear Regression" = "LR",
    "GLM"               = "GLM",
    "Random Forest"     = "RF",
    "SVM"               = "SVM",
    "GBM"               = "GBM",
    "MARS"              = "MARS",
    "Neural Network"    = "NNET",
    "PLSR"              = "PLSR",
    "XGBoost"           = "XGBoost",
    "LightGBM"          = "LightGBM",
    "BART"              = "BART",
    "EBM"               = "EBM",
    "Stacking"          = "Stacking"
  )
  model_key <- unname(perf_to_key[best_model_name])
  if (is.na(model_key) || !model_key %in% names(models)) {
    model_key <- best_model_name  # 兜底：直接用性能表名称
  }

  cat("模型键值:", model_key, "\n\n")

  # 初始化重要性数据框
  importance_df <- NULL
  pdp_data <- NULL

  # =============================================================================
  # 根据模型类型提取特征重要性和PDP
  # =============================================================================

  if (model_key == "BART") {
    # -------------------- BART模型 --------------------
    cat("提取BART特征重要性...\n")

    bart_model <- models$BART
    var_counts <- bart_model$varcount
    var_usage <- colMeans(var_counts)

    importance_df <- data.frame(
      Feature = feature_names,
      Importance = var_usage,
      stringsAsFactors = FALSE
    )

    importance_df <- importance_df[order(-importance_df$Importance), ]
    rownames(importance_df) <- NULL
    importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

    write.csv(importance_df,
              sprintf("%s_Feature_Importance.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

    cat("特征重要性排名:\n")
    print(importance_df)
    cat("\n")

    # 计算BART部分依赖图（优化版本：并行计算）
    cat("计算BART部分依赖图（并行加速版本）...\n")

    top_features <- head(importance_df$Feature, 6)

    n_grid <- 20
    sample_size <- min(30, nrow(X_train))

    n_cores <- parallel::detectCores()
    use_cores <- max(1, min(n_cores - 1, 4))

    cat(sprintf("  使用%d个核心并行计算，%d个代表性样本，%d个网格点\n",
                use_cores, sample_size, n_grid))

    set.seed(123)
    sample_idx <- sample(1:nrow(X_train), sample_size)
    X_sample <- X_train[sample_idx, ]

    compute_pdp_for_feature <- function(feat) {
      feat_idx <- which(feature_names == feat)
      feat_values <- seq(min(X_train[, feat_idx]), max(X_train[, feat_idx]),
                         length.out = n_grid)
      pd_values <- numeric(length(feat_values))
      for (j in 1:length(feat_values)) {
        X_temp <- as.matrix(X_sample)
        X_temp[, feat_idx] <- feat_values[j]
        pred_temp <- predict(bart_model, X_temp)
        pd_values[j] <- mean(pred_temp)
      }
      data.frame(
        Feature = rep(feat, n_grid),
        x = feat_values,
        y = pd_values,
        stringsAsFactors = FALSE
      )
    }

    if (.Platform$OS.type == "windows") {
      cl <- parallel::makeCluster(use_cores)
      parallel::clusterEvalQ(cl, library(BART))
      parallel::clusterExport(cl, c("bart_model", "X_train", "X_sample",
                                     "feature_names", "n_grid"),
                             envir = environment())
      pdp_list <- parallel::parLapply(cl, top_features, compute_pdp_for_feature)
      parallel::stopCluster(cl)
    } else {
      pdp_list <- parallel::mclapply(top_features, compute_pdp_for_feature,
                                     mc.cores = use_cores)
    }

    pdp_data <- do.call(rbind, pdp_list)
    write.csv(pdp_data,
              sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

  } else if (model_key == "EBM") {
    # -------------------- EBM模型 --------------------
    cat("提取EBM特征重要性...\n")

    feature_importances <- as.numeric(ebm_global$data()$scores)

    importance_df <- data.frame(
      Feature = feature_names,
      Importance = feature_importances,
      stringsAsFactors = FALSE
    )

    importance_df <- importance_df[order(-importance_df$Importance), ]
    rownames(importance_df) <- NULL
    importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

    write.csv(importance_df,
              sprintf("%s_Feature_Importance.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

    cat("特征重要性排名:\n")
    print(importance_df)
    cat("\n")

    cat("使用Python提取部分依赖数据...\n")

    py_run_string("
import numpy as np
import pandas as pd

pdp_all_data = []

for i in range(len(r.feature_names)):
    feature_name = r.feature_names[i]
    try:
        feature_data = r.ebm_global.data(i)
        x_values = feature_data['names']
        y_values = feature_data['scores']

        for x, y in zip(x_values, y_values):
            pdp_all_data.append({
                'Feature': feature_name,
                'x': float(x),
                'y': float(y)
            })
    except Exception as e:
        print(f'Error extracting {feature_name}: {str(e)}')

pdp_df = pd.DataFrame(pdp_all_data)
pdp_df.to_csv('ebm_pdp_temp.csv', index=False)
print(f'Extracted {len(pdp_df)} PDP data points')
")

    pdp_data <- read.csv("ebm_pdp_temp.csv", stringsAsFactors = FALSE)
    write.csv(pdp_data,
              sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

  } else if (model_key == "RF") {
    # -------------------- 随机森林模型 --------------------
    cat("提取随机森林特征重要性...\n")

    rf_model <- models$RF$finalModel
    var_imp <- importance(rf_model)

    importance_df <- data.frame(
      Feature = rownames(var_imp),
      Importance = var_imp[, 1],
      stringsAsFactors = FALSE
    )

    importance_df <- importance_df[order(-importance_df$Importance), ]
    rownames(importance_df) <- NULL
    importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

    write.csv(importance_df,
              sprintf("%s_Feature_Importance.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

    cat("特征重要性排名:\n")
    print(importance_df)
    cat("\n")

    cat("计算随机森林部分依赖图...\n")

    suppressPackageStartupMessages(library(pdp))

    top_features <- head(importance_df$Feature, 6)
    pdp_all_data <- list()

    for (feat in top_features) {
      cat("  计算", feat, "的PDP...\n")
      pd_result <- partial(
        models$RF,
        pred.var = feat,
        train = train_data,
        grid.resolution = 50
      )
      for (i in 1:nrow(pd_result)) {
        pdp_all_data[[length(pdp_all_data) + 1]] <- data.frame(
          Feature = feat,
          x = pd_result[i, 1],
          y = pd_result$yhat[i],
          stringsAsFactors = FALSE
        )
      }
    }

    pdp_data <- do.call(rbind, pdp_all_data)
    write.csv(pdp_data,
              sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

  } else if (model_key == "XGBoost") {
    # -------------------- XGBoost模型 --------------------
    cat("提取XGBoost特征重要性...\n")

    xgb_imp <- varImp(models$XGBoost)$importance

    importance_df <- data.frame(
      Feature = rownames(xgb_imp),
      Importance = xgb_imp$Overall,
      stringsAsFactors = FALSE
    )

    importance_df <- importance_df[order(-importance_df$Importance), ]
    rownames(importance_df) <- NULL
    importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

    write.csv(importance_df,
              sprintf("%s_Feature_Importance.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

    cat("特征重要性排名:\n")
    print(importance_df)
    cat("\n")

    cat("计算XGBoost部分依赖图...\n")

    suppressPackageStartupMessages(library(pdp))

    top_features <- head(importance_df$Feature, 6)
    pdp_all_data <- list()

    for (feat in top_features) {
      cat("  计算", feat, "的PDP...\n")
      pd_result <- partial(
        models$XGBoost,
        pred.var = feat,
        train = train_data,
        grid.resolution = 50
      )
      for (i in 1:nrow(pd_result)) {
        pdp_all_data[[length(pdp_all_data) + 1]] <- data.frame(
          Feature = feat,
          x = pd_result[i, 1],
          y = pd_result$yhat[i],
          stringsAsFactors = FALSE
        )
      }
    }

    pdp_data <- do.call(rbind, pdp_all_data)
    write.csv(pdp_data,
              sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

  } else if (model_key == "LightGBM") {
    # -------------------- LightGBM模型 --------------------
    cat("提取LightGBM特征重要性...\n")

    lgb_model <- models$LightGBM
    lgb_imp <- lgb.importance(lgb_model, percentage = TRUE)

    importance_df <- data.frame(
      Feature = lgb_imp$Feature,
      Importance = lgb_imp$Gain,
      stringsAsFactors = FALSE
    )

    importance_df <- importance_df[order(-importance_df$Importance), ]
    rownames(importance_df) <- NULL
    importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

    write.csv(importance_df,
              sprintf("%s_Feature_Importance.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

    cat("特征重要性排名:\n")
    print(importance_df)
    cat("\n")

    cat("计算LightGBM部分依赖图（手动方法）...\n")

    top_features <- head(importance_df$Feature, 6)
    pdp_all_data <- list()

    for (feat in top_features) {
      cat("  计算", feat, "的PDP...\n")
      feat_values <- sort(unique(quantile(X_train[[feat]], probs = seq(0.05, 0.95, length.out = 50))))
      pd_vals <- numeric(length(feat_values))

      for (j in seq_along(feat_values)) {
        X_temp <- X_train
        X_temp[[feat]] <- feat_values[j]
        pd_vals[j] <- mean(predict(lgb_model, as.matrix(X_temp)))
      }

      for (i in seq_along(feat_values)) {
        pdp_all_data[[length(pdp_all_data) + 1]] <- data.frame(
          Feature = feat,
          x = feat_values[i],
          y = pd_vals[i],
          stringsAsFactors = FALSE
        )
      }
    }

    pdp_data <- do.call(rbind, pdp_all_data)
    write.csv(pdp_data,
              sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

  } else {
    # -------------------- 其他模型（使用通用varImp） --------------------
    cat("提取", best_model_name, "特征重要性（通用方法）...\n")

    tryCatch({
      var_imp <- varImp(models[[model_key]])

      if (is.list(var_imp) && "importance" %in% names(var_imp)) {
        imp_matrix <- var_imp$importance
      } else {
        imp_matrix <- var_imp
      }

      if (ncol(imp_matrix) == 1) {
        importance_values <- imp_matrix[, 1]
      } else {
        importance_values <- rowMeans(imp_matrix)
      }

      importance_df <- data.frame(
        Feature = rownames(imp_matrix),
        Importance = importance_values,
        stringsAsFactors = FALSE
      )

      importance_df <- importance_df[order(-importance_df$Importance), ]
      rownames(importance_df) <- NULL
      importance_df$Relative_Importance <- importance_df$Importance / max(importance_df$Importance)

      write.csv(importance_df,
                sprintf("%s_Feature_Importance.csv", model_rank_prefix),
                row.names = FALSE)
      cat(sprintf("✓ %s_Feature_Importance.csv\n\n", model_rank_prefix))

      cat("特征重要性排名:\n")
      print(importance_df)
      cat("\n")

      cat("计算部分依赖图...\n")

      suppressPackageStartupMessages(library(pdp))

      top_features <- head(importance_df$Feature, 6)
      pdp_all_data <- list()

      for (feat in top_features) {
        cat("  计算", feat, "的PDP...\n")
        pd_result <- partial(
          models[[model_key]],
          pred.var = feat,
          train = train_data,
          grid.resolution = 50
        )
        for (i in 1:nrow(pd_result)) {
          pdp_all_data[[length(pdp_all_data) + 1]] <- data.frame(
            Feature = feat,
            x = pd_result[i, 1],
            y = pd_result$yhat[i],
            stringsAsFactors = FALSE
          )
        }
      }

      pdp_data <- do.call(rbind, pdp_all_data)
      write.csv(pdp_data,
                sprintf("%s_Partial_Dependence_Data.csv", model_rank_prefix),
                row.names = FALSE)
      cat(sprintf("✓ %s_Partial_Dependence_Data.csv\n\n", model_rank_prefix))

    }, error = function(e) {
      cat("无法提取该模型的特征重要性和PDP\n")
      cat("错误信息:", e$message, "\n\n")
    })
  }

  # =============================================================================
  # 绘图：特征重要性（绝对值）
  # =============================================================================

  if (!is.null(importance_df)) {
    cat("生成特征重要性柱状图（绝对值，PDF）...\n")

    pdf(sprintf("Figure_%s_Feature_Importance.pdf", model_rank_prefix),
        width = 8, height = 6)

    par(mar = c(5, 10, 3, 2))

    imp_vals <- importance_df$Importance
    imp_max  <- max(imp_vals)

    barplot(
      rev(imp_vals),
      names.arg = rev(importance_df$Feature),
      horiz = TRUE,
      las = 1,
      col = "steelblue",
      main = paste(best_model_name, "- Feature Importance"),
      xlab = "Importance",
      xlim = c(0, imp_max * 1.15),
      cex.names = 1.2,
      cex.axis = 1.1,
      cex.lab = 1.2,
      cex.main = 1.3
    )

    text(
      x = rev(imp_vals) + imp_max * 0.03,
      y = (1:nrow(importance_df)) * 1.2 - 0.5,
      labels = round(rev(imp_vals), 3),
      cex = 1,
      pos = 4
    )

    abline(v = seq(0, imp_max, length.out = 6), col = "gray90", lty = 2)

    dev.off()
    cat(sprintf("✓ Figure_%s_Feature_Importance.pdf\n\n", model_rank_prefix))
  }

  # =============================================================================
  # 绘图：PDP图
  # =============================================================================

  if (!is.null(pdp_data)) {
    cat("生成论文风格PDP图（PDF）...\n")

    top_features <- head(importance_df$Feature, 6)

    pdp_top6 <- pdp_data[pdp_data$Feature %in% top_features, ]

    pdp_normalized <- do.call(rbind, lapply(top_features, function(feat) {
      feat_data <- pdp_top6[pdp_top6$Feature == feat, ]
      x_min <- min(feat_data$x)
      x_max <- max(feat_data$x)
      x_normalized <- (feat_data$x - x_min) / (x_max - x_min)
      data.frame(
        Feature = feat,
        x_original = feat_data$x,
        x_normalized = x_normalized,
        y = feat_data$y,
        stringsAsFactors = FALSE
      )
    }))

    write.csv(pdp_normalized,
              sprintf("%s_PDP_Normalized_Top6.csv", model_rank_prefix),
              row.names = FALSE)
    cat(sprintf("✓ %s_PDP_Normalized_Top6.csv\n\n", model_rank_prefix))

    # 基础R版本PDP图（PDF）
    pdf(sprintf("Figure_%s_PDP_Paper_Style.pdf", model_rank_prefix),
        width = 10, height = 6)

    par(mar = c(5, 5, 3, 2))

    colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")

    y_range <- range(pdp_normalized$y)
    y_margin <- (y_range[2] - y_range[1]) * 0.1

    plot(NULL,
         xlim = c(0, 1),
         ylim = c(y_range[1] - y_margin, y_range[2] + y_margin),
         xlab = "Normalized Feature Value",
         ylab = "Partial Dependence",
         xaxt = "n",
         cex.lab = 1.3,
         cex.axis = 1.2,
         las = 1)

    axis(1, at = c(0, 1), labels = c("Low", "High"), cex.axis = 1.3, font = 2)

    grid(col = "gray80", lty = 2)

    for (i in 1:length(top_features)) {
      feat <- top_features[i]
      feat_data <- pdp_normalized[pdp_normalized$Feature == feat, ]
      feat_data <- feat_data[order(feat_data$x_normalized), ]
      lines(feat_data$x_normalized, feat_data$y,
            col = colors[i], lwd = 2.5, lty = 1)
    }

    legend("topright",
           legend = paste0(top_features, " (",
                          round(importance_df$Importance[1:length(top_features)], 3), ")"),
           col = colors[1:length(top_features)],
           lty = 1,
           lwd = 2.5,
           cex = 1,
           title = "Feature (Importance)",
           bty = "n")

    if (y_range[1] < 0 && y_range[2] > 0) {
      abline(h = 0, col = "gray50", lty = 2, lwd = 1)
    }

    dev.off()
    cat(sprintf("✓ Figure_%s_PDP_Paper_Style.pdf\n\n", model_rank_prefix))

    # ggplot2版本PDP图（PDF）
    cat("生成ggplot2美化版PDP图（PDF）...\n")

    pdp_normalized$Feature <- factor(pdp_normalized$Feature, levels = top_features)

    p <- ggplot(pdp_normalized, aes(x = x_normalized, y = y,
                                    color = Feature, group = Feature)) +
      geom_line(linewidth = 1.2) +
      scale_color_manual(
        values = colors[1:length(top_features)],
        labels = paste0(top_features, " (",
                       round(importance_df$Importance[1:length(top_features)], 3), ")")
      ) +
      scale_x_continuous(
        breaks = c(0, 1),
        labels = c("Low", "High"),
        limits = c(0, 1)
      ) +
      labs(
        x = "Normalized Feature Value",
        y = "Partial Dependence",
        color = "Feature\n(Importance)"
      ) +
      theme_bw() +
      theme(
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11),
        axis.text.x = element_text(size = 12, face = "bold"),
        legend.title = element_text(size = 11, face = "bold"),
        legend.text = element_text(size = 10),
        legend.position = "right",
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
      )

    ggsave(sprintf("Figure_%s_PDP_ggplot.pdf", model_rank_prefix),
           p, width = 10, height = 6)
    cat(sprintf("✓ Figure_%s_PDP_ggplot.pdf\n\n", model_rank_prefix))
  }

  # =============================================================================
  # 绘制全数据散点图（训练集/测试集着色）
  # =============================================================================

  cat("绘制全数据观测值vs预测值图（训练集/测试集着色）...\n")

  tryCatch({
    best_model <- models[[model_key]]

    if (model_key == "BART") {
      bart_pred_matrix <- predict(best_model, as.matrix(X))
      all_predictions <- colMeans(bart_pred_matrix)
    } else if (model_key == "LightGBM") {
      all_predictions <- predict(best_model, as.matrix(X))
    } else if (model_key == "EBM") {
      all_predictions <- best_model$predict(as.matrix(X))
    } else {
      all_predictions <- predict(best_model, data.frame(X))
    }

    dataset_labels <- rep("Training", nrow(data))
    dataset_labels[-train_index] <- "Test"

    plot_data <- data.frame(
      Observed  = y,
      Predicted = as.numeric(all_predictions),
      Dataset   = factor(dataset_labels, levels = c("Training", "Test")),
      stringsAsFactors = FALSE
    )

    n_train <- sum(plot_data$Dataset == "Training")
    n_test  <- sum(plot_data$Dataset == "Test")
    cat(sprintf("  训练集: %d, 测试集: %d\n", n_train, n_test))

    test_pred <- predictions[[model_key]]
    if (is.list(test_pred)) test_pred <- unlist(test_pred)
    test_pred <- as.numeric(test_pred)
    test_r2   <- calc_R2(test_pred, y_test)
    test_rmse <- Metrics::rmse(test_pred, y_test)

    dataset_colors <- c(Training = "#0066CC", Test = "#CC0000")

    # 基础R版本
    pdf(sprintf("Figure_%s_All_Data_Scatter.pdf", model_rank_prefix),
        width = 7, height = 6)

    par(mar = c(5, 5, 4, 2))

    plot(plot_data$Observed, plot_data$Predicted,
         pch = 19,
         col = dataset_colors[as.character(plot_data$Dataset)],
         cex = 1.0,
         xlab = "Observed Q10",
         ylab = "Predicted Q10",
         main = sprintf("%s\nTest R² = %.4f, RMSE = %.4f",
                        best_model_name, test_r2, test_rmse),
         cex.lab = 1.2,
         cex.axis = 1.1,
         cex.main = 1.1)

    abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
    # 添加回归线
    fit_line <- lm(Predicted ~ Observed, data = plot_data)
    abline(fit_line, col = "blue", lwd = 2)
    grid(col = "gray85", lty = 2)

    legend("topleft",
           legend = c(sprintf("Training (n=%d)", n_train),
                      sprintf("Test (n=%d)",     n_test)),
           col    = dataset_colors,
           pch    = 19,
           pt.cex = 1.2,
           cex    = 0.95,
           bty    = "n")

    dev.off()
    cat(sprintf("✓ Figure_%s_All_Data_Scatter.pdf\n\n", model_rank_prefix))

    # ggplot2版本
    cat("生成ggplot2版本的全数据散点图...\n")

    p_all <- ggplot(plot_data, aes(x = Observed, y = Predicted, color = Dataset,shape=Dataset)) +
      geom_point(size = 2.5, alpha = 0.75) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
      geom_smooth(method = "lm", se = FALSE, color = "blue", linewidth = 1,
                  formula = y ~ x) +
      scale_color_manual(
        values = dataset_colors,
        labels = c(sprintf("Training (n=%d)", n_train),
                   sprintf("Test (n=%d)",     n_test))
      ) +
      scale_shape_manual(
        values = c("Training" = 16, "Test" = 17),
        labels = c(paste0("Training (n=", sum(plot_data$Dataset == "Training"), ")"),
                   paste0("Test (n=", sum(plot_data$Dataset == "Test"), ")"))
      ) +
      labs(
        title    = best_model_name,
        subtitle = sprintf("Test R² = %.4f    RMSE = %.4f", test_r2, test_rmse),
        x        = "Observed Q10",
        y        = "Predicted Q10",
        color    = NULL
      ) +
      theme_bw() +
      theme(
        plot.title    = element_text(size = 13, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
        axis.title    = element_text(size = 12, face = "bold"),
        axis.text     = element_text(size = 11),
        legend.text   = element_text(size = 10),
        legend.position   = "top",
        panel.grid.minor  = element_blank(),
        panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.8)
      )

    ggsave(sprintf("Figure_%s_All_Data_Scatter_ggplot.pdf", model_rank_prefix),
           p_all, width = 7, height = 6)
    cat(sprintf("✓ Figure_%s_All_Data_Scatter_ggplot.pdf\n\n", model_rank_prefix))

  }, error = function(e) {
    cat("绘制全数据散点图时出错:", e$message, "\n\n")
  })

}  # end for (rank_i in 1:min(2, nrow(performance)))

# ============================================================================
# 第六部分：保存工作空间和生成报告
# ============================================================================

cat("=" , rep("=", 68), "\n", sep="")
cat("第六部分：保存工作空间和生成最终报告\n")
cat("=" , rep("=", 68), "\n\n", sep="")

# 保存工作空间
save.image("Q10_Complete_Analysis_Workspace.RData")
cat("✓ Q10_Complete_Analysis_Workspace.RData\n\n")

# 生成最终报告
cat("============================================================\n")
cat("完整分析流程完成！\n")
cat("============================================================\n\n")

cat("一、模型训练结果\n")
cat("  成功训练:", model_count, "/13 个模型\n\n")

cat("二、最佳模型（Top 3）:\n")
for (i in 1:min(3, nrow(performance))) {
  cat(sprintf("  %d. %s\n", i, performance$Model[i]))
  cat(sprintf("     R²   = %.4f\n", performance$R2[i]))
  cat(sprintf("     RMSE = %.4f\n", performance$RMSE[i]))
  cat(sprintf("     MAE  = %.4f\n\n", performance$MAE[i]))
}

cat("三、前两位模型可解释性文件（以 Rank1_/Rank2_ 前缀区分）:\n\n")

cat("  数据文件:\n")
cat("    - Model_Performance_Final.csv (模型性能表)\n")
cat("    - Model_Predictions_Final.csv (预测值)\n")
cat("    - Rank1_<模型名>_Feature_Importance.csv\n")
cat("    - Rank2_<模型名>_Feature_Importance.csv\n")
cat("    - Rank1_<模型名>_Partial_Dependence_Data.csv\n")
cat("    - Rank2_<模型名>_Partial_Dependence_Data.csv\n")
cat("    - Rank1_<模型名>_PDP_Normalized_Top6.csv\n")
cat("    - Rank2_<模型名>_PDP_Normalized_Top6.csv\n\n")

cat("  PDF图表:\n")
cat("    - Model_Performance_Circles.pdf (性能对比圆圈图)\n")
cat("    - Observed_vs_Predicted_All_Models.pdf (散点图)\n")
cat("    - Figure_Rank1_<模型名>_Feature_Importance.pdf (第1名特征重要性，绝对值)\n")
cat("    - Figure_Rank2_<模型名>_Feature_Importance.pdf (第2名特征重要性，绝对值)\n")
cat("    - Figure_Rank1_<模型名>_PDP_Paper_Style.pdf\n")
cat("    - Figure_Rank2_<模型名>_PDP_Paper_Style.pdf\n")
cat("    - Figure_Rank1_<模型名>_PDP_ggplot.pdf\n")
cat("    - Figure_Rank2_<模型名>_PDP_ggplot.pdf\n")
cat("    - Figure_Rank1_<模型名>_All_Data_Scatter.pdf\n")
cat("    - Figure_Rank2_<模型名>_All_Data_Scatter.pdf\n")
cat("    - Figure_Rank1_<模型名>_All_Data_Scatter_ggplot.pdf\n")
cat("    - Figure_Rank2_<模型名>_All_Data_Scatter_ggplot.pdf\n\n")

cat("  中间数据:\n")
cat("    - intermediate_model_01_LR.rds ~ intermediate_model_13_Stacking.rds\n")
cat("    - Q10_Complete_Analysis_Workspace.RData (完整工作空间)\n\n")

cat("保存位置: D:/Claude/Q10/model3_DOM\n\n")
cat("分析完成！所有图表已保存为PDF格式。\n")
cat("第1名模型:", performance$Model[1], "\n")
cat("第2名模型:", performance$Model[2], "\n")
