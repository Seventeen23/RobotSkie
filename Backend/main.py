"""
Dota 2 Match Predictor — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import json
from pathlib import Path

from data_loader import build_inference_vector

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
META_PATH  = MODELS_DIR / "metadata.json"

MODELS_DIRS = [BASE_DIR / "models", MODELS_DIR]

# ── Global state ──────────────────────────────────────────────
models   = {}
metadata = {}


def load_models():
    global models, metadata

    for models_dir in MODELS_DIRS:
        meta_path = models_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            print(f"✓ Metadata loaded from {models_dir.name} — {len(metadata.get('feature_columns', []))} features")
            break
    else:
        print("⚠️  No metadata.json found — run train_models.py first.")
        metadata = {"model_performance": {}, "feature_columns": [],
                    "radiant_team_classes": [], "dire_team_classes": [],
                    "use_team_features": False, "total_heroes": 150}

    for models_dir in MODELS_DIRS:
        rf_path = models_dir / "random_forest.joblib"
        if rf_path.exists() and "random_forest" not in models:
            models["random_forest"] = joblib.load(rf_path)
            print(f"✓ Random Forest loaded from {models_dir.name}")

        xgb_path = models_dir / "xgboost.json"
        if xgb_path.exists() and "xgboost" not in models:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(xgb_path))
            models["xgboost"] = m
            print(f"✓ XGBoost loaded from {models_dir.name}")

        nn_path = models_dir / "neural_network.joblib"
        if nn_path.exists() and "neural_network" not in models:
            models["neural_network"] = joblib.load(nn_path)
            print(f"✓ Neural Network loaded from {models_dir.name}")

        lgb_path = models_dir / "lightgbm.joblib"
        if lgb_path.exists() and "lightgbm" not in models:
            import lightgbm as lgb
            models["lightgbm"] = joblib.load(lgb_path)
            print(f"✓ LightGBM loaded from {models_dir.name}")

    if not models:
        print("\n⚠️  No trained models — demo mode active.")


# ── Pydantic Schemas ──────────────────────────────────────────
class PredictionRequest(BaseModel):
    radiant_heroes: List[int] = Field(..., description="5 hero IDs for Radiant")
    dire_heroes:    List[int] = Field(..., description="5 hero IDs for Dire")
    banned_heroes:  List[int] = Field(default=[], description="Banned hero IDs (0–10)")
    radiant_team:   Optional[str] = Field(default=None, description="Radiant team name (optional)")
    dire_team:      Optional[str] = Field(default=None, description="Dire team name (optional)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "radiant_heroes": [74, 106, 91, 97, 67],
                "dire_heroes":    [14, 41, 86, 1, 110],
                "banned_heroes":  [8, 11, 35],
                "radiant_team":   "Team Spirit",
                "dire_team":      "Tundra Esports",
            }
        }
    }


class ModelPrediction(BaseModel):
    model_name:              str
    display_name:            str
    radiant_win_probability: float
    dire_win_probability:    float
    predicted_winner:        str
    confidence:              str
    accuracy:                Optional[float] = None
    auc:                     Optional[float] = None
    description:             str


class PredictionResponse(BaseModel):
    radiant_heroes:       List[int]
    dire_heroes:          List[int]
    banned_heroes:        List[int]
    radiant_team:         Optional[str]
    dire_team:            Optional[str]
    predictions:          List[ModelPrediction]
    ensemble_radiant_prob: float
    ensemble_winner:      str


# ── Helpers ────────────────────────────────────────────────────
MODEL_INFO = {
    "random_forest": {
        "display_name": "Random Forest",
        "description": (
            "300 decision trees trained on hero picks, bans, and team identity. "
            "Captures non-linear synergies and counter-pick patterns with high interpretability."
        ),
    },
    "xgboost": {
        "display_name": "XGBoost",
        "description": (
            "Gradient-boosted trees (up to 500 estimators, early stopping). "
            "Excels at learning draft meta patterns and team-specific win tendencies."
        ),
    },
    "neural_network": {
        "display_name": "Neural Network",
        "description": (
            "4-layer MLP (512→256→128→64) trained on pick/ban vectors + team one-hot encoding. "
            "Learns deep compositional hero embeddings across pro league games."
        ),
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "description": (
            "Gradient boosting with leaf-wise tree growth. Fast training with competitive "
            "accuracy on draft patterns and hero synergy features."
        ),
    },
}


def get_confidence_label(prob: float) -> str:
    diff = abs(prob - 0.5)
    if   diff > 0.25: return "High"
    elif diff > 0.15: return "Medium"
    elif diff > 0.08: return "Low"
    else:             return "Toss-up"


def mock_predict(radiant: List[int], dire: List[int], model_key: str) -> float:
    """Deterministic demo prediction when models are not yet trained."""
    offsets = {"random_forest": 0.0, "xgboost": 0.03, "neural_network": -0.02, "lightgbm": 0.01}
    seed = (sum(radiant) * 7 - sum(dire) * 3) & 0xFFFFFFFF
    np.random.seed(seed)
    base = float(np.clip(np.random.normal(0.5, 0.12), 0.15, 0.85))
    return float(np.clip(base + offsets.get(model_key, 0), 0.05, 0.95))


# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Dota 2 Match Predictor API",
    description="Predicts Dota 2 pro match outcomes using 4 ML models trained on 2023–2026 data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://robot-skie.vercel.app"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Dota 2 Match Predictor API",
        "docs": "/docs",
        "models_loaded": list(models.keys()),
        "status": "ready"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(models),
        "model_names": list(models.keys())
    }


@app.get("/models")
async def list_models():
    perf = metadata.get("model_performance", {})
    model_list = [
        {
            "id": key,
            "display_name": info["display_name"],
            "description": info["description"],
            "loaded": key in models,
            "accuracy": perf.get(key, {}).get("accuracy"),
            "auc":      perf.get(key, {}).get("auc"),
        }
        for key, info in MODEL_INFO.items()
    ]
    return {
        "models": model_list,
        "training_samples":  metadata.get("training_samples"),
        "use_team_features": metadata.get("use_team_features", False),
        "feature_count":     len(metadata.get("feature_columns", [])),
    }


@app.get("/teams")
async def get_teams():
    """Return all known team names from training data."""
    return {
        "radiant_teams": sorted(metadata.get("radiant_team_classes", [])),
        "dire_teams":    sorted(metadata.get("dire_team_classes",    [])),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    # Validate picks
    if len(req.radiant_heroes) != 5:
        raise HTTPException(status_code=400, detail="radiant_heroes must have exactly 5 heroes")
    if len(req.dire_heroes) != 5:
        raise HTTPException(status_code=400, detail="dire_heroes must have exactly 5 heroes")

    overlap = set(req.radiant_heroes) & set(req.dire_heroes)
    if overlap:
        raise HTTPException(status_code=400, detail=f"Hero IDs on both teams: {sorted(overlap)}")

    # Build feature vector using the same pipeline as training
    if metadata.get("feature_columns"):
        features = build_inference_vector(
            radiant_picks=req.radiant_heroes,
            dire_picks=req.dire_heroes,
            banned_heroes=req.banned_heroes,
            radiant_team=req.radiant_team,
            dire_team=req.dire_team,
            model_meta=metadata,
        )
    else:
        # Fallback: simple pick vector if no metadata yet
        vec = np.zeros(150, dtype=np.float32)
        for h in req.radiant_heroes:
            if 0 < h < 150: vec[h] =  1.0
        for h in req.dire_heroes:
            if 0 < h < 150: vec[h] = -1.0
        features = vec.reshape(1, -1)

    perf = metadata.get("model_performance", {})
    predictions = []

    for model_key, info in MODEL_INFO.items():
        if model_key in models:
            try:
                proba = float(models[model_key].predict_proba(features)[0][1])
            except Exception as e:
                print(f"⚠️ {model_key} predict error: {e}")
                proba = mock_predict(req.radiant_heroes, req.dire_heroes, model_key)
        else:
            proba = mock_predict(req.radiant_heroes, req.dire_heroes, model_key)

        predictions.append(ModelPrediction(
            model_name=model_key,
            display_name=info["display_name"],
            radiant_win_probability=round(proba, 4),
            dire_win_probability=round(1 - proba, 4),
            predicted_winner="Radiant" if proba > 0.5 else "Dire",
            confidence=get_confidence_label(proba),
            accuracy=perf.get(model_key, {}).get("accuracy"),
            auc=perf.get(model_key, {}).get("auc"),
            description=info["description"],
        ))

    ensemble_prob = float(np.mean([p.radiant_win_probability for p in predictions]))

    return PredictionResponse(
        radiant_heroes=req.radiant_heroes,
        dire_heroes=req.dire_heroes,
        banned_heroes=req.banned_heroes,
        radiant_team=req.radiant_team,
        dire_team=req.dire_team,
        predictions=predictions,
        ensemble_radiant_prob=round(ensemble_prob, 4),
        ensemble_winner="Radiant" if ensemble_prob > 0.5 else "Dire",
    )


@app.get("/heroes")
async def get_heroes():
    """Returns all Dota 2 heroes with IDs and names."""
    return {"heroes": HERO_DATA}


# ── Hero Data ─────────────────────────────────────────────────
HERO_DATA = [
    {"id": 1, "name": "Anti-Mage", "roles": ["Carry", "Escape"], "attr": "agi"},
    {"id": 2, "name": "Axe", "roles": ["Initiator", "Durable"], "attr": "str"},
    {"id": 3, "name": "Bane", "roles": ["Support", "Disabler"], "attr": "int"},
    {"id": 4, "name": "Bloodseeker", "roles": ["Carry", "Jungler"], "attr": "agi"},
    {"id": 5, "name": "Crystal Maiden", "roles": ["Support", "Disabler"], "attr": "int"},
    {"id": 6, "name": "Drow Ranger", "roles": ["Carry", "Disabler"], "attr": "agi"},
    {"id": 7, "name": "Earthshaker", "roles": ["Initiator", "Disabler", "Support"], "attr": "str"},
    {"id": 8, "name": "Juggernaut", "roles": ["Carry", "Pusher"], "attr": "agi"},
    {"id": 9, "name": "Mirana", "roles": ["Carry", "Support", "Escape"], "attr": "agi"},
    {"id": 10, "name": "Morphling", "roles": ["Carry", "Escape"], "attr": "agi"},
    {"id": 11, "name": "Shadow Fiend", "roles": ["Carry", "Nuker"], "attr": "agi"},
    {"id": 12, "name": "Phantom Lancer", "roles": ["Carry", "Escape", "Pusher"], "attr": "agi"},
    {"id": 13, "name": "Puck", "roles": ["Escape", "Nuker", "Disabler"], "attr": "int"},
    {"id": 14, "name": "Pudge", "roles": ["Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 15, "name": "Razor", "roles": ["Carry", "Durable"], "attr": "agi"},
    {"id": 16, "name": "Sand King", "roles": ["Initiator", "Disabler", "Support", "Nuker"], "attr": "str"},
    {"id": 17, "name": "Storm Spirit", "roles": ["Carry", "Escape", "Nuker", "Initiator"], "attr": "int"},
    {"id": 18, "name": "Sven", "roles": ["Carry", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 19, "name": "Tiny", "roles": ["Carry", "Initiator", "Nuker", "Pusher"], "attr": "str"},
    {"id": 20, "name": "Vengeful Spirit", "roles": ["Support", "Disabler", "Initiator"], "attr": "agi"},
    {"id": 21, "name": "Windranger", "roles": ["Carry", "Support", "Disabler", "Nuker", "Escape"], "attr": "int"},
    {"id": 22, "name": "Zeus", "roles": ["Nuker"], "attr": "int"},
    {"id": 23, "name": "Kunkka", "roles": ["Carry", "Support", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 25, "name": "Lina", "roles": ["Carry", "Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 26, "name": "Lion", "roles": ["Support", "Disabler", "Nuker"], "attr": "int"},
    {"id": 27, "name": "Shadow Shaman", "roles": ["Support", "Pusher", "Disabler"], "attr": "int"},
    {"id": 28, "name": "Slardar", "roles": ["Carry", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 29, "name": "Tidehunter", "roles": ["Initiator", "Disabler", "Durable", "Support"], "attr": "str"},
    {"id": 30, "name": "Witch Doctor", "roles": ["Support", "Disabler", "Nuker"], "attr": "int"},
    {"id": 31, "name": "Lich", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 32, "name": "Riki", "roles": ["Carry", "Escape", "Nuker"], "attr": "agi"},
    {"id": 33, "name": "Enigma", "roles": ["Initiator", "Disabler", "Pusher", "Jungler"], "attr": "int"},
    {"id": 34, "name": "Tinker", "roles": ["Carry", "Nuker", "Pusher"], "attr": "int"},
    {"id": 35, "name": "Sniper", "roles": ["Carry", "Nuker"], "attr": "agi"},
    {"id": 36, "name": "Necrophos", "roles": ["Carry", "Nuker", "Durable"], "attr": "int"},
    {"id": 37, "name": "Warlock", "roles": ["Support", "Disabler", "Initiator", "Nuker"], "attr": "int"},
    {"id": 38, "name": "Beastmaster", "roles": ["Initiator", "Disabler", "Pusher", "Durable"], "attr": "str"},
    {"id": 39, "name": "Queen of Pain", "roles": ["Carry", "Escape", "Nuker", "Disabler"], "attr": "int"},
    {"id": 40, "name": "Venomancer", "roles": ["Carry", "Pusher", "Nuker"], "attr": "agi"},
    {"id": 41, "name": "Faceless Void", "roles": ["Carry", "Initiator"], "attr": "agi"},
    {"id": 42, "name": "Wraith King", "roles": ["Carry", "Durable", "Initiator", "Disabler"], "attr": "str"},
    {"id": 43, "name": "Death Prophet", "roles": ["Carry", "Nuker", "Pusher"], "attr": "int"},
    {"id": 44, "name": "Phantom Assassin", "roles": ["Carry", "Escape"], "attr": "agi"},
    {"id": 45, "name": "Pugna", "roles": ["Nuker", "Pusher", "Support"], "attr": "int"},
    {"id": 46, "name": "Templar Assassin", "roles": ["Carry", "Escape", "Nuker"], "attr": "agi"},
    {"id": 47, "name": "Viper", "roles": ["Carry", "Nuker", "Durable"], "attr": "agi"},
    {"id": 48, "name": "Luna", "roles": ["Carry", "Pusher"], "attr": "agi"},
    {"id": 49, "name": "Dragon Knight", "roles": ["Carry", "Durable", "Initiator", "Pusher"], "attr": "str"},
    {"id": 50, "name": "Dazzle", "roles": ["Support", "Healer"], "attr": "int"},
    {"id": 51, "name": "Clockwerk", "roles": ["Initiator", "Durable", "Disabler"], "attr": "str"},
    {"id": 52, "name": "Leshrac", "roles": ["Carry", "Nuker", "Pusher", "Disabler"], "attr": "int"},
    {"id": 53, "name": "Nature's Prophet", "roles": ["Carry", "Pusher", "Jungler"], "attr": "int"},
    {"id": 54, "name": "Lifestealer", "roles": ["Carry", "Durable", "Jungler"], "attr": "str"},
    {"id": 55, "name": "Dark Seer", "roles": ["Initiator", "Disabler", "Nuker", "Jungler"], "attr": "int"},
    {"id": 56, "name": "Clinkz", "roles": ["Carry", "Escape", "Nuker"], "attr": "agi"},
    {"id": 57, "name": "Omniknight", "roles": ["Support", "Durable"], "attr": "str"},
    {"id": 58, "name": "Enchantress", "roles": ["Support", "Jungler", "Pusher", "Nuker"], "attr": "int"},
    {"id": 59, "name": "Huskar", "roles": ["Carry", "Durable", "Initiator", "Nuker"], "attr": "str"},
    {"id": 60, "name": "Night Stalker", "roles": ["Carry", "Durable", "Disabler", "Initiator"], "attr": "str"},
    {"id": 61, "name": "Broodmother", "roles": ["Carry", "Escape", "Pusher", "Jungler"], "attr": "agi"},
    {"id": 62, "name": "Bounty Hunter", "roles": ["Carry", "Escape", "Nuker", "Support"], "attr": "agi"},
    {"id": 63, "name": "Weaver", "roles": ["Carry", "Escape", "Nuker"], "attr": "agi"},
    {"id": 64, "name": "Jakiro", "roles": ["Support", "Disabler", "Nuker", "Pusher"], "attr": "int"},
    {"id": 65, "name": "Batrider", "roles": ["Initiator", "Disabler", "Nuker", "Escape"], "attr": "int"},
    {"id": 66, "name": "Chen", "roles": ["Support", "Pusher", "Jungler"], "attr": "int"},
    {"id": 67, "name": "Spectre", "roles": ["Carry", "Escape", "Durable"], "attr": "agi"},
    {"id": 68, "name": "Ancient Apparition", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 69, "name": "Doom", "roles": ["Carry", "Durable", "Disabler", "Nuker", "Jungler"], "attr": "str"},
    {"id": 70, "name": "Ursa", "roles": ["Carry", "Jungler", "Initiator"], "attr": "agi"},
    {"id": 71, "name": "Spirit Breaker", "roles": ["Carry", "Disabler", "Initiator", "Durable", "Escape"], "attr": "str"},
    {"id": 72, "name": "Gyrocopter", "roles": ["Carry", "Nuker"], "attr": "agi"},
    {"id": 73, "name": "Alchemist", "roles": ["Carry", "Durable", "Disabler", "Jungler", "Nuker"], "attr": "str"},
    {"id": 74, "name": "Invoker", "roles": ["Carry", "Nuker", "Disabler", "Escape", "Pusher"], "attr": "int"},
    {"id": 75, "name": "Silencer", "roles": ["Carry", "Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 76, "name": "Outworld Devourer", "roles": ["Carry", "Nuker", "Disabler"], "attr": "int"},
    {"id": 77, "name": "Lycan", "roles": ["Carry", "Jungler", "Pusher", "Initiator"], "attr": "str"},
    {"id": 78, "name": "Brewmaster", "roles": ["Initiator", "Durable", "Nuker", "Disabler"], "attr": "str"},
    {"id": 79, "name": "Shadow Demon", "roles": ["Support", "Disabler", "Nuker", "Escape"], "attr": "int"},
    {"id": 80, "name": "Lone Druid", "roles": ["Carry", "Pusher", "Jungler", "Durable"], "attr": "agi"},
    {"id": 81, "name": "Chaos Knight", "roles": ["Carry", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 82, "name": "Meepo", "roles": ["Carry", "Pusher", "Escape", "Nuker"], "attr": "agi"},
    {"id": 83, "name": "Treant Protector", "roles": ["Support", "Initiator", "Durable", "Disabler"], "attr": "str"},
    {"id": 84, "name": "Ogre Magi", "roles": ["Support", "Durable", "Nuker", "Disabler"], "attr": "str"},
    {"id": 85, "name": "Undying", "roles": ["Support", "Durable", "Initiator", "Nuker"], "attr": "str"},
    {"id": 86, "name": "Rubick", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 87, "name": "Disruptor", "roles": ["Support", "Nuker", "Disabler", "Initiator"], "attr": "int"},
    {"id": 88, "name": "Nyx Assassin", "roles": ["Support", "Initiator", "Escape", "Nuker", "Disabler"], "attr": "agi"},
    {"id": 89, "name": "Naga Siren", "roles": ["Carry", "Support", "Disabler", "Escape", "Pusher"], "attr": "agi"},
    {"id": 90, "name": "Keeper of the Light", "roles": ["Support", "Nuker", "Pusher", "Disabler"], "attr": "int"},
    {"id": 91, "name": "Io", "roles": ["Support", "Escape"], "attr": "str"},
    {"id": 92, "name": "Visage", "roles": ["Carry", "Support", "Nuker", "Disabler", "Durable"], "attr": "int"},
    {"id": 93, "name": "Slark", "roles": ["Carry", "Escape"], "attr": "agi"},
    {"id": 94, "name": "Medusa", "roles": ["Carry", "Pusher", "Durable"], "attr": "agi"},
    {"id": 95, "name": "Troll Warlord", "roles": ["Carry", "Disabler"], "attr": "agi"},
    {"id": 96, "name": "Centaur Warrunner", "roles": ["Initiator", "Durable", "Disabler", "Carry"], "attr": "str"},
    {"id": 97, "name": "Magnus", "roles": ["Initiator", "Disabler", "Nuker", "Carry"], "attr": "str"},
    {"id": 98, "name": "Timbersaw", "roles": ["Carry", "Nuker", "Escape", "Durable"], "attr": "str"},
    {"id": 99, "name": "Bristleback", "roles": ["Carry", "Durable", "Nuker"], "attr": "str"},
    {"id": 100, "name": "Tusk", "roles": ["Support", "Initiator", "Disabler", "Carry"], "attr": "str"},
    {"id": 101, "name": "Skywrath Mage", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 102, "name": "Abaddon", "roles": ["Carry", "Support", "Durable"], "attr": "str"},
    {"id": 103, "name": "Elder Titan", "roles": ["Support", "Initiator", "Disabler", "Nuker"], "attr": "str"},
    {"id": 104, "name": "Legion Commander", "roles": ["Carry", "Durable", "Initiator", "Disabler"], "attr": "str"},
    {"id": 105, "name": "Techies", "roles": ["Support", "Disabler", "Nuker"], "attr": "int"},
    {"id": 106, "name": "Ember Spirit", "roles": ["Carry", "Escape", "Nuker", "Initiator", "Disabler"], "attr": "agi"},
    {"id": 107, "name": "Earth Spirit", "roles": ["Support", "Escape", "Nuker", "Initiator", "Disabler"], "attr": "str"},
    {"id": 108, "name": "Underlord", "roles": ["Carry", "Initiator", "Disabler", "Durable", "Nuker"], "attr": "str"},
    {"id": 109, "name": "Terrorblade", "roles": ["Carry", "Escape", "Pusher"], "attr": "agi"},
    {"id": 110, "name": "Phoenix", "roles": ["Support", "Nuker", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 111, "name": "Oracle", "roles": ["Support", "Escape"], "attr": "int"},
    {"id": 112, "name": "Winter Wyvern", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 113, "name": "Arc Warden", "roles": ["Carry", "Pusher", "Escape", "Nuker"], "attr": "agi"},
    {"id": 114, "name": "Monkey King", "roles": ["Carry", "Escape", "Initiator", "Disabler", "Nuker"], "attr": "agi"},
    {"id": 119, "name": "Dark Willow", "roles": ["Support", "Escape", "Nuker", "Disabler"], "attr": "int"},
    {"id": 120, "name": "Pangolier", "roles": ["Carry", "Escape", "Disabler", "Initiator", "Nuker", "Durable"], "attr": "agi"},
    {"id": 121, "name": "Grimstroke", "roles": ["Support", "Nuker", "Disabler"], "attr": "int"},
    {"id": 123, "name": "Hoodwink", "roles": ["Support", "Escape", "Nuker", "Disabler"], "attr": "agi"},
    {"id": 126, "name": "Void Spirit", "roles": ["Carry", "Escape", "Nuker", "Disabler", "Initiator"], "attr": "int"},
    {"id": 128, "name": "Snapfire", "roles": ["Support", "Nuker", "Disabler", "Initiator"], "attr": "str"},
    {"id": 129, "name": "Mars", "roles": ["Carry", "Initiator", "Disabler", "Nuker", "Durable"], "attr": "str"},
    {"id": 135, "name": "Dawnbreaker", "roles": ["Carry", "Support", "Durable", "Initiator", "Nuker"], "attr": "str"},
    {"id": 136, "name": "Marci", "roles": ["Carry", "Support", "Disabler", "Initiator", "Durable"], "attr": "str"},
    {"id": 137, "name": "Primal Beast", "roles": ["Carry", "Durable", "Initiator", "Disabler", "Nuker"], "attr": "str"},
    {"id": 138, "name": "Muerta", "roles": ["Carry", "Support", "Disabler", "Nuker"], "attr": "int"},
    {"id": 139, "name": "Ringmaster", "roles": ["Support", "Disabler", "Escape", "Nuker"], "attr": "int"},
    {"id": 140, "name": "Kez", "roles": ["Carry", "Escape", "Disabler"], "attr": "agi"},
    {"id": 141, "name": "Largo", "roles": ["Support", "Disabler", "Durable"], "attr": "str"},
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
