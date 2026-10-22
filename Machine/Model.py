import xgboost as xg
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

train = pd.read_csv ("train_master.csv")
validate = pd.read_csv("validation_master.csv")

target = "fire_next_7d"

exclude = {
    target,
    "grid_id",
    "prediction_date",
    "weather_station_id",
    "station_id"
}

features = [
    columns for columns in train.select_dtypes(include = "number").columns
    if columns not in exclude and columns in validate.columns
]

X_train, y_train, X_validate, y_validate = train[features], train[target], validate[features], validate[target]

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=50,
    n_jobs=4,
    random_state=42,
)

model.fit(
    X_train,
    y_validate,
    eval_set=[(X_validate, y_validate)],
    verbose=100,
)

probabilities = model.predict_proba(X_validate)[:, 1]

print("Features:", features)
print("Validation prevalence:", y_validate.mean())
print("Average precision:", average_precision_score(y_validate, probabilities))
print("ROC-AUC:", roc_auc_score(y_validate, probabilities))

model.save_model("C:\Users\tanne\wildfire detector\Wildfire-Predictor\Machine")

