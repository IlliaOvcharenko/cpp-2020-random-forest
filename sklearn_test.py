import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


data_folder = Path("data")
tr_df = pd.read_csv(data_folder / "train.csv")
val_df = pd.read_csv(data_folder / "val.csv")

models = {
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest_10": RandomForestClassifier(n_estimators=10, random_state=42),
    "random_forest_20": RandomForestClassifier(n_estimators=20, random_state=42),
    "xgboost": XGBClassifier(random_state=42),
}


for model_name, model in models.items():

    model.fit(tr_df.iloc[:, :-1], tr_df["y"])

    tr_preds = model.predict(tr_df.iloc[:, :-1])
    tr_score = accuracy_score(tr_df["y"], tr_preds)

    val_preds = model.predict(val_df.iloc[:, :-1])
    val_score = accuracy_score(val_df["y"], val_preds)

    print(f"model: {model_name}")
    print(f"train score: {tr_score}, val score: {val_score}")
    print()
