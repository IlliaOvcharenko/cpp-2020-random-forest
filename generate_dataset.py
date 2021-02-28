import random

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm

np.random.seed(42)
random.seed(42)

n_classes = 4
n_examples = 500
n_features = 50

df = pd.DataFrame()
for cls_id in tqdm(range(n_classes)):
    class_mu = np.random.uniform(0.1, 0.9)
    class_sigma = np.random.uniform(0.1, 0.2)

    for exmpl_id in range(n_examples):
        features = [False] * n_features
        n_filled = int(np.clip(np.random.normal(0.5, 0.2), 0, 1) * (n_features-1))
        for _ in range(n_filled):
            feature_idx = np.clip(np.random.normal(class_mu, class_sigma), 0, 1)
            feature_idx = int(feature_idx * (n_features-1))
            features[feature_idx] = True
        features = {f"x_{i}": v for i, v in enumerate(features)}
        features["y"] = cls_id
        df = df.append(features, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

data_folder = Path("data")

tr_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])
tr_df.to_csv(data_folder / "train.csv", index=False)
val_df.to_csv(data_folder / "val.csv", index=False)

# model = DecisionTreeClassifier(random_state=42)
# # model = RandomForestClassifier(n_estimators=10, random_state=42)
# # model = RandomForestClassifier(n_estimators=20, random_state=42)
# # model = XGBClassifier(random_state=42)
#
# model.fit(tr_df.iloc[:, :-1], tr_df["y"])
#
# tr_preds = model.predict(tr_df.iloc[:, :-1])
# tr_score = accuracy_score(tr_df["y"], tr_preds)
#
# val_preds = model.predict(val_df.iloc[:, :-1])
# val_score = accuracy_score(val_df["y"], val_preds)
#
# print(f"train score: {tr_score}, val score: {val_score}")
