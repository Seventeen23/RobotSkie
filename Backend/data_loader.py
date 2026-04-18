"""
data_loader.py
──────────────
Builds training features from 3 CSVs:
  1. main_metadata  — match result (radiant_win), patch, duration
  2. picks_bans     — hero picks and bans per match (is_pick, hero_id, team, order)
  3. match_teams    — team names per match (radiant.name, dire.name)

Feature vector per match:
  [0..139]   hero pick encoding  — +1 radiant pick, -1 dire pick  (140 dims)
  [140..279] hero ban encoding   — +1 if hero was banned           (140 dims)
  [280..N]   team one-hot        — radiant_team_id, dire_team_id   (2 * n_teams dims, optional)

Total without team features: 280 dims
Total with    team features: 280 + 2*n_teams dims
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
TOTAL_HEROES = 150  # hero IDs go up to ~136 but leave headroom


# ── CSV detection ──────────────────────────────────────────────────────────────

def _detect_csv(files: list[Path], required_cols: list[str]) -> Path | None:
    """Return the first CSV whose columns contain all required_cols."""
    for f in files:
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()
            if all(c in header for c in required_cols):
                return f
        except Exception:
            pass
    return None


def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Auto-detect and load the 3 required CSVs from DATA_DIR.
    Returns (main_meta, picks_bans, match_teams).
    Raises FileNotFoundError with a clear message if any are missing.
    """
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. "
            "Place your 4 CSVs there and run again."
        )

    main_meta = _detect_csv(files, ["match_id", "radiant_win", "duration"])
    picks_bans = _detect_csv(files, ["match_id", "is_pick", "hero_id", "team", "order"])
    match_teams = _detect_csv(files, ["match_id", "radiant.name", "dire.name"])

    missing = []
    if main_meta is None:
        missing.append("main_metadata (needs: match_id, radiant_win, duration)")
    if picks_bans is None:
        missing.append("picks_bans (needs: match_id, is_pick, hero_id, team, order)")
    if match_teams is None:
        missing.append("match_teams (needs: match_id, radiant.name, dire.name)")

    if missing:
        raise FileNotFoundError(
            "Could not auto-detect the following CSVs:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + f"\n\nFiles found in {DATA_DIR}:\n"
            + "\n".join(f"  {f.name}" for f in files)
        )

    print(f"  ✓ main_metadata : {main_meta.name}")
    print(f"  ✓ picks_bans    : {picks_bans.name}")
    print(f"  ✓ match_teams   : {match_teams.name}")

    df_meta   = pd.read_csv(main_meta,   low_memory=False)
    df_picks  = pd.read_csv(picks_bans,  low_memory=False)
    df_teams  = pd.read_csv(match_teams, low_memory=False)

    print(f"\n  Rows — meta: {len(df_meta):,}  picks_bans: {len(df_picks):,}  teams: {len(df_teams):,}")
    return df_meta, df_picks, df_teams


# ── Pick/ban pivot ─────────────────────────────────────────────────────────────

def build_pick_ban_matrix(df_picks: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot picks_bans into one row per match_id with columns:
      pick_r_{hero_id}  — 1 if hero was picked by radiant
      pick_d_{hero_id}  — 1 if hero was picked by dire
      ban_{hero_id}     — 1 if hero was banned (either team)
    Then compute the asymmetric combined pick vector (+1/-1/0).
    """
    # Normalise boolean column (may be True/False or 1/0 or "True"/"False")
    df_picks = df_picks.copy()
    df_picks["is_pick"] = df_picks["is_pick"].astype(str).str.lower().isin(["true", "1"])
    df_picks["hero_id"] = pd.to_numeric(df_picks["hero_id"], errors="coerce")
    df_picks["team"]    = pd.to_numeric(df_picks["team"],    errors="coerce")
    df_picks = df_picks.dropna(subset=["match_id", "hero_id", "team"])
    df_picks["hero_id"] = df_picks["hero_id"].astype(int)
    df_picks["team"]    = df_picks["team"].astype(int)

    # team == 0 → radiant, team == 1 → dire  (Dota 2 convention)
    picks_only = df_picks[df_picks["is_pick"]]
    bans_only  = df_picks[~df_picks["is_pick"]]

    match_ids = df_picks["match_id"].unique()
    n = len(match_ids)
    idx_map = {mid: i for i, mid in enumerate(match_ids)}

    # Arrays: hero picks (+1 radiant, -1 dire), hero bans (+1 banned)
    pick_vec = np.zeros((n, TOTAL_HEROES), dtype=np.float32)
    ban_vec  = np.zeros((n, TOTAL_HEROES), dtype=np.float32)

    for row in picks_only.itertuples(index=False):
        mid  = row.match_id
        hid  = row.hero_id
        team = row.team
        if mid not in idx_map or hid <= 0 or hid >= TOTAL_HEROES:
            continue
        i = idx_map[mid]
        pick_vec[i, hid] = 1.0 if team == 0 else -1.0

    for row in bans_only.itertuples(index=False):
        mid = row.match_id
        hid = row.hero_id
        if mid not in idx_map or hid <= 0 or hid >= TOTAL_HEROES:
            continue
        ban_vec[idx_map[mid], hid] = 1.0

    # Build column names
    pick_cols = [f"pick_{h}" for h in range(TOTAL_HEROES)]
    ban_cols  = [f"ban_{h}"  for h in range(TOTAL_HEROES)]

    df_out = pd.DataFrame(
        np.hstack([pick_vec, ban_vec]),
        columns=pick_cols + ban_cols
    )
    df_out.insert(0, "match_id", match_ids)
    return df_out


# ── Team encoding ──────────────────────────────────────────────────────────────

def build_team_features(
    df_teams: pd.DataFrame,
    match_ids: np.ndarray,
    min_appearances: int = 10,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    One-hot encode radiant/dire team names.
    Teams with fewer than `min_appearances` matches are grouped as '__other__'.
    Returns (feature_df, radiant_encoder, dire_encoder).
    """
    df_teams = df_teams.copy()
    df_teams["radiant.name"] = df_teams["radiant.name"].fillna("__unknown__").str.strip()
    df_teams["dire.name"]    = df_teams["dire.name"].fillna("__unknown__").str.strip()

    # Collapse rare teams
    for col in ["radiant.name", "dire.name"]:
        counts = df_teams[col].value_counts()
        rare   = counts[counts < min_appearances].index
        df_teams[col] = df_teams[col].where(~df_teams[col].isin(rare), "__other__")

    # Fit encoders on full set, then align to match_ids
    r_enc = LabelEncoder().fit(df_teams["radiant.name"])
    d_enc = LabelEncoder().fit(df_teams["dire.name"])

    # Merge onto requested match_ids
    base = pd.DataFrame({"match_id": match_ids})
    merged = base.merge(
        df_teams[["match_id", "radiant.name", "dire.name"]],
        on="match_id", how="left"
    )
    merged["radiant.name"] = merged["radiant.name"].fillna("__unknown__")
    merged["dire.name"]    = merged["dire.name"].fillna("__unknown__")

    # Handle unseen labels gracefully
    def safe_transform(enc: LabelEncoder, series: pd.Series) -> np.ndarray:
        known = set(enc.classes_)
        mapped = series.apply(lambda x: x if x in known else "__other__")
        if "__other__" not in known:
            # add __other__ class dynamically
            enc.classes_ = np.append(enc.classes_, "__other__")
        return enc.transform(mapped)

    r_codes = safe_transform(r_enc, merged["radiant.name"])
    d_codes = safe_transform(d_enc, merged["dire.name"])

    n_r = len(r_enc.classes_)
    n_d = len(d_enc.classes_)

    r_ohe = np.eye(n_r, dtype=np.float32)[r_codes]
    d_ohe = np.eye(n_d, dtype=np.float32)[d_codes]

    r_cols = [f"r_team_{c}" for c in r_enc.classes_]
    d_cols = [f"d_team_{c}" for c in d_enc.classes_]

    df_feat = pd.DataFrame(
        np.hstack([r_ohe, d_ohe]),
        columns=r_cols + d_cols
    )
    df_feat.insert(0, "match_id", match_ids)

    print(f"  Team features: {n_r} radiant teams + {n_d} dire teams = {n_r+n_d} cols")
    return df_feat, r_enc, d_enc


# ── Master builder ─────────────────────────────────────────────────────────────

def build_dataset(
    use_team_features: bool = True,
    min_team_appearances: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Full pipeline: load → join → featurise.

    Returns
    -------
    X         : float32 feature matrix  (n_matches × n_features)
    y         : int array               (n_matches,)  1=radiant win
    feat_cols : list of feature column names
    meta      : dict with encoder info for inference
    """
    print("📂 Loading CSVs...")
    df_meta, df_picks, df_teams = load_csvs()

    # ── Target ────────────────────────────────────────────────
    df_meta = df_meta[["match_id", "radiant_win"]].dropna()
    df_meta["radiant_win"] = (
        df_meta["radiant_win"]
        .astype(str).str.lower()
        .isin(["true", "1"])
        .astype(int)
    )

    # ── Pick/ban features ─────────────────────────────────────
    print("\n🎯 Building pick/ban feature matrix...")
    df_pb = build_pick_ban_matrix(df_picks)

    # ── Join meta → pick/ban ──────────────────────────────────
    df = df_pb.merge(df_meta, on="match_id", how="inner")
    print(f"  Matches after join with meta: {len(df):,}")

    feat_cols = [c for c in df.columns if c.startswith("pick_") or c.startswith("ban_")]

    # ── Team features ──────────────────────────────────────────
    r_enc = d_enc = None
    if use_team_features:
        print("\n🏆 Building team features...")
        df_tf, r_enc, d_enc = build_team_features(
            df_teams,
            df["match_id"].values,
            min_appearances=min_team_appearances,
        )
        team_cols = [c for c in df_tf.columns if c != "match_id"]
        df = df.merge(df_tf, on="match_id", how="left")
        # Fill any unmatched matches with 0
        df[team_cols] = df[team_cols].fillna(0)
        feat_cols = feat_cols + team_cols

    y = df["radiant_win"].values.astype(int)
    X = df[feat_cols].values.astype(np.float32)

    print(f"\n✅ Final dataset: {X.shape[0]:,} matches × {X.shape[1]} features")
    print(f"   Radiant wins: {y.sum():,} / {len(y):,}  ({y.mean()*100:.1f}%)")

    meta = {
        "total_heroes": TOTAL_HEROES,
        "use_team_features": use_team_features,
        "feature_columns": feat_cols,
        "n_pick_features": TOTAL_HEROES * 2,   # pick_vec + ban_vec
        "radiant_team_classes": r_enc.classes_.tolist() if r_enc else [],
        "dire_team_classes":    d_enc.classes_.tolist() if d_enc else [],
    }

    return X, y, feat_cols, meta


# ── Inference helper (used by FastAPI) ────────────────────────────────────────

def build_inference_vector(
    radiant_picks: list[int],
    dire_picks: list[int],
    banned_heroes: list[int],
    radiant_team: str | None,
    dire_team: str | None,
    model_meta: dict,
) -> np.ndarray:
    """
    Build a single inference feature vector matching what the models were trained on.
    """
    feat_cols: list[str] = model_meta["feature_columns"]
    vec = np.zeros(len(feat_cols), dtype=np.float32)
    col_idx = {c: i for i, c in enumerate(feat_cols)}

    for hid in radiant_picks:
        key = f"pick_{hid}"
        if key in col_idx:
            vec[col_idx[key]] = 1.0

    for hid in dire_picks:
        key = f"pick_{hid}"
        if key in col_idx:
            vec[col_idx[key]] = -1.0

    for hid in banned_heroes:
        key = f"ban_{hid}"
        if key in col_idx:
            vec[col_idx[key]] = 1.0

    if model_meta.get("use_team_features"):
        r_classes = model_meta.get("radiant_team_classes", [])
        d_classes = model_meta.get("dire_team_classes", [])

        r_name = radiant_team if radiant_team in r_classes else "__other__"
        d_name = dire_team    if dire_team    in d_classes else "__other__"

        r_key = f"r_team_{r_name}"
        d_key = f"d_team_{d_name}"
        if r_key in col_idx:
            vec[col_idx[r_key]] = 1.0
        if d_key in col_idx:
            vec[col_idx[d_key]] = 1.0

    return vec.reshape(1, -1)
