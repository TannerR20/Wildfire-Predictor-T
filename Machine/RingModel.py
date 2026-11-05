from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

train = pd.read_csv("train_master.csv")
validate = pd.read_csv("validation_master.csv")

target = "fire_next_7d"

exclude = {
    target,
    "grid_id",
    "prediction_date",
    "weather_station_id",
    "station_id",
}

# Replace these with your actual ring feature columns.
ring_features = {
    1: ["population_density_k1", "forest_pct_k1"],
    2: ["population_density_k2", "forest_pct_k2"],
    3: ["population_density_k3", "forest_pct_k3"],
}

all_ring_features = [
    column
    for columns in ring_features.values()
    for column in columns
]

# Fail clearly if the configured ring columns are missing.
for name, data in [("train", train), ("validation", validate)]:
    missing = set(all_ring_features) - set(data.columns)
    if missing:
        raise ValueError(f"{name} is missing ring columns: {sorted(missing)}")

# Local baseline excludes ALL configured ring features.
local_features = [
    column
    for column in train.select_dtypes(include="number").columns
    if column not in exclude
    and column not in all_ring_features
    and column in validate.columns
]

experiments = {"local": local_features}

for k in sorted(ring_features):
    experiments[f"through_k{k}"] = local_features + [
        column
        for ring in sorted(ring_features)
        if ring <= k
        for column in ring_features[ring]
    ]

output_dir = Path(
    r"C:\Users\tanne\wildfire detector\Wildfire-Predictor\Machine"
)
output_dir.mkdir(parents=True, exist_ok=True)

y_train = train[target]
y_validate = validate[target]

results = []

for name, features in experiments.items():
    X_train = train[features]
    X_validate = validate[features]

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

    print(f"\nTraining: {name} ({len(features)} features)")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validate, y_validate)],
        verbose=100,
    )

    probabilities = model.predict_proba(X_validate)[:, 1]

    results.append({
        "experiment": name,
        "feature_count": len(features),
        "best_iteration": model.best_iteration,
        "validation_prevalence": y_validate.mean(),
        "average_precision": average_precision_score(
            y_validate, probabilities
        ),
        "roc_auc": roc_auc_score(y_validate, probabilities),
    })

    model.save_model(output_dir / f"{name}.json")

    pd.Series(features).to_json(
        output_dir / f"{name}_features.json",
        orient="values",
    )

results = pd.DataFrame(results)
results.to_csv(output_dir / "ablation_results.csv", index=False)

print("\nAblation results:")
print(results.to_string(index=False))