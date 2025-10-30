from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

@dataclass
class FeatureSpace:
    num_cols: list
    cat_cols: list

def make_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))])
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("No features provided")
    return ColumnTransformer(transformers)

def make_logreg(num_cols, cat_cols, C=1.0, max_iter=2000):
    pre = make_preprocessor(num_cols, cat_cols)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="saga", n_jobs=-1, penalty="l2", class_weight=None, verbose=0)
    return Pipeline([("pre", pre), ("clf", clf)])

def make_xgb(num_cols, cat_cols, params: Dict[str, Any]):
    pre = make_preprocessor(num_cols, cat_cols)
    clf = XGBClassifier(**params)
    return Pipeline([("pre", pre), ("clf", clf)])

def make_catboost(num_cols, cat_cols, params: Dict[str, Any]):
    pre = make_preprocessor(num_cols, cat_cols)
    clf = CatBoostClassifier(**params)
    return Pipeline([("pre", pre), ("clf", clf)])

def stack_predictions(preds: np.ndarray) -> np.ndarray:
    return preds.mean(axis=0)
