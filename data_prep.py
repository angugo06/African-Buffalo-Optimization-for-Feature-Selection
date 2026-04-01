import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_prepare(csv_path: str, target_col: str, test_size: float = 0.3, random_state: Optional[int] = 42):
    df = pd.read_csv(csv_path)
    assert target_col in df.columns, f"Target column '{target_col}' not found."
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_tf = Pipeline([("scaler", StandardScaler())])
    categorical_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    ct = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ], remainder="drop")

    Xtr_df, Xva_df, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    Xt = ct.fit_transform(Xtr_df)
    Xv = ct.transform(Xva_df)

    try:
        feat_names = ct.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]

    return Xt, Xv, ytr, yva, feat_names
