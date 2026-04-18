"""
train_models.py
───────────────
Train 3 Dota 2 match-outcome prediction models using:
  • Hero picks  (+1 radiant / -1 dire asymmetric encoding)
  • Hero bans   (+1 if banned)
  • Team identity (one-hot, teams with <10 matches → '__other__')

Models:
  1. Random Forest
  2. XGBoost
  3. Neural Network (MLP)
  4. LightGBM

Run:
  python train_models.py
  python train_models.py --no-teams      # skip team features
  python train_models.py --min-team 20   # raise rarity threshold
"""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from data_loader import build_dataset

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_all(X_train, X_test, y_train, y_test) -> dict:
    results = {}

    # ── 1. Random Forest ──────────────────────────────────────
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred  = (rf_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, rf_pred)
    auc = roc_auc_score(y_test, rf_proba)
    print(f"   Accuracy: {acc:.4f}   AUC-ROC: {auc:.4f}")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    results["random_forest"] = {"accuracy": acc, "auc": auc}

    # ── 2. XGBoost ────────────────────────────────────────────
    print("\n⚡ Training XGBoost...")
    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred  = (xgb_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, xgb_pred)
    auc = roc_auc_score(y_test, xgb_proba)
    print(f"   Accuracy: {acc:.4f}   AUC-ROC: {auc:.4f}")
    print(f"   Best iteration: {xgb_model.best_iteration}")
    xgb_model.save_model(str(MODELS_DIR / "xgboost.json"))
    results["xgboost"] = {"accuracy": acc, "auc": auc}

    # ── 3. Neural Network ─────────────────────────────────────
    print("\n🧠 Training Neural Network (MLP)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0005,
        batch_size=256,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    mlp_proba = mlp.predict_proba(X_test)[:, 1]
    mlp_pred  = (mlp_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, mlp_pred)
    auc = roc_auc_score(y_test, mlp_proba)
    print(f"   Accuracy: {acc:.4f}   AUC-ROC: {auc:.4f}")
    print(f"   Iterations run: {mlp.n_iter_}")
    joblib.dump(mlp, MODELS_DIR / "neural_network.joblib")
    results["neural_network"] = {"accuracy": acc, "auc": auc}

        # ── 4. LightGBM ───────────────────────────────────────────
    print("\n💡 Training LightGBM...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.01,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        class_weight="balanced",
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
    )

    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred  = (lgb_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, lgb_pred)
    auc = roc_auc_score(y_test, lgb_proba)

    print(f"   Accuracy: {acc:.4f}   AUC-ROC: {auc:.4f}")

    joblib.dump(lgb_model, MODELS_DIR / "lightgbm.joblib")

    results["lightgbm"] = {
        "accuracy": acc,
        "auc": auc
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Dota 2 match prediction models")
    parser.add_argument("--no-teams",  action="store_true", help="Exclude team identity features")
    parser.add_argument("--min-team",  type=int, default=10, help="Min matches for a team to get its own column")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction (default 0.15)")
    args = parser.parse_args()

    print("=" * 62)
    print("  Dota 2 Oracle — Model Training")
    print("=" * 62)

    X, y, feat_cols, data_meta = build_dataset(
        use_team_features=not args.no_teams,
        min_team_appearances=args.min_team,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    print(f"\n  Train: {len(X_train):,}   Test: {len(X_test):,}")

    perf = train_all(X_train, X_test, y_train, y_test)

    meta = {
        **data_meta,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "model_performance": perf,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 62)
    print("  Results")
    print("=" * 62)
    for name, m in perf.items():
        bar = "█" * int(m["auc"] * 40) + "░" * (40 - int(m["auc"] * 40))
        print(f"  {name:<20s}  Acc {m['accuracy']*100:5.1f}%  AUC {m['auc']:.4f}  {bar}")
    print()


if __name__ == "__main__":
    main()
