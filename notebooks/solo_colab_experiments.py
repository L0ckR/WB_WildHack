#!/usr/bin/env python3
# %%
"""
Colab-ready workflow for the WB WildHack solo track.

Run this file as a notebook-style script in Colab or Jupyter.
It extends the baseline with:
1. large EDA,
2. feature-engineered tabular models,
3. optional zero-shot foundation-model forecasts,
4. validation-calibrated blending.
"""

# %%
# %pip install -q pandas pyarrow numpy matplotlib seaborn scikit-learn xgboost catboost torch transformers accelerate

# %%
from __future__ import annotations

import gc
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.base import clone
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (14, 5)


# %%
@dataclass
class Config:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts_solo")
    train_path: str = "train_solo_track.parquet"
    test_path: str = "test_solo_track.parquet"
    target_col: str = "target_1h"
    forecast_points: int = 8
    freq_minutes: int = 30
    min_history: int = 96
    valid_days: int = 7
    train_days: int = 28
    holdout_anchor_limit: int = 96
    lags: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 24, 48)
    windows: Tuple[int, ...] = (3, 6, 12, 24, 48)
    catboost_iterations: int = 2500
    xgb_estimators: int = 1200
    xgb_learning_rate: float = 0.03
    run_timesfm: bool = True
    run_timer: bool = True
    random_state: int = 42

    @property
    def future_target_cols(self) -> List[str]:
        return [f"target_step_{step}" for step in range(1, self.forecast_points + 1)]


CFG = Config()
CFG.artifacts_dir.mkdir(exist_ok=True, parents=True)
CFG.data_dir.mkdir(exist_ok=True, parents=True)


# %%
class WapePlusRbias:
    @property
    def name(self) -> str:
        return "wape_plus_rbias"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        denom = np.clip(y_true.sum(), 1e-8, None)
        wape = np.abs(y_pred - y_true).sum() / denom
        rbias = abs(y_pred.sum() / denom - 1.0)
        return float(wape + rbias)


METRIC = WapePlusRbias()


# %%
def download_solo_data(cfg: Config) -> None:
    import zipfile
    import urllib.request

    url = "https://static-basket-03.wbbasket.ru/vol48/mlcontests/wruuQNI2.zip"
    zip_path = cfg.data_dir / "solo_track.zip"
    train_path = cfg.data_dir / cfg.train_path
    test_path = cfg.data_dir / cfg.test_path

    if train_path.exists() and test_path.exists():
        print("Solo data already present.")
        return

    print("Downloading solo archive...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.data_dir)
    print("Extracted:", sorted(p.name for p in cfg.data_dir.glob("*.parquet")))


# %%
def load_solo_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(cfg.data_dir / cfg.train_path)
    test_df = pd.read_parquet(cfg.data_dir / cfg.test_path)

    for frame in (train_df, test_df):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame.sort_values(["route_id", "timestamp"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    return train_df, test_df


# %%
def infer_status_cols(df: pd.DataFrame) -> List[str]:
    return sorted(col for col in df.columns if col.startswith("status_"))


def safe_float32(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df


def add_future_targets(df: pd.DataFrame, target_col: str, future_cols: Sequence[str]) -> pd.DataFrame:
    route_group = df.groupby("route_id", sort=False)
    for step, col in enumerate(future_cols, start=1):
        df[col] = route_group[target_col].shift(-step)
    return df


# %%
def run_large_eda(train_df: pd.DataFrame, status_cols: Sequence[str], cfg: Config) -> None:
    out_dir = cfg.artifacts_dir / "eda"
    out_dir.mkdir(exist_ok=True, parents=True)

    overview = pd.DataFrame(
        {
            "dtype": train_df.dtypes.astype(str),
            "missing_cnt": train_df.isna().sum(),
            "missing_pct": (train_df.isna().mean() * 100).round(4),
            "n_unique": train_df.nunique(dropna=False),
        }
    ).sort_values(["missing_pct", "n_unique"], ascending=[False, False])
    overview.to_csv(out_dir / "overview.csv")

    route_stats = (
        train_df.groupby("route_id")
        .agg(
            rows=("route_id", "size"),
            target_mean=(cfg.target_col, "mean"),
            target_std=(cfg.target_col, "std"),
            target_zero_share=(cfg.target_col, lambda s: float((s == 0).mean())),
            target_p95=(cfg.target_col, lambda s: float(s.quantile(0.95))),
        )
        .sort_values("target_mean", ascending=False)
    )
    route_stats.to_csv(out_dir / "route_stats.csv")

    temporal = train_df[["timestamp", cfg.target_col]].copy()
    temporal["date"] = temporal["timestamp"].dt.date
    temporal["hour"] = temporal["timestamp"].dt.hour
    temporal["dayofweek"] = temporal["timestamp"].dt.dayofweek

    daily = temporal.groupby("date")[cfg.target_col].sum().reset_index()
    hourly = temporal.groupby("hour")[cfg.target_col].mean().reset_index()
    dow = temporal.groupby("dayofweek")[cfg.target_col].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.histplot(train_df[cfg.target_col].clip(upper=train_df[cfg.target_col].quantile(0.995)), bins=80, ax=axes[0, 0])
    axes[0, 0].set_title("Target distribution clipped at p99.5")

    sns.lineplot(data=daily, x="date", y=cfg.target_col, ax=axes[0, 1])
    axes[0, 1].set_title("Daily target volume")
    axes[0, 1].tick_params(axis="x", rotation=45)

    sns.lineplot(data=hourly, x="hour", y=cfg.target_col, marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("Average target by hour")

    sns.barplot(data=dow, x="dayofweek", y=cfg.target_col, ax=axes[1, 1])
    axes[1, 1].set_title("Average target by weekday")

    plt.tight_layout()
    plt.savefig(out_dir / "target_temporal_patterns.png", dpi=160)
    plt.close(fig)

    if status_cols:
        clipped = train_df[status_cols + [cfg.target_col]].copy()
        for col in status_cols + [cfg.target_col]:
            clipped[col] = clipped[col].clip(upper=clipped[col].quantile(0.99))

        corr = clipped.corr(numeric_only=True)
        plt.figure(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.7)))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Status / target correlations")
        plt.tight_layout()
        plt.savefig(out_dir / "status_target_corr.png", dpi=160)
        plt.close()

    route_zero = route_stats["target_zero_share"]
    plt.figure(figsize=(12, 5))
    sns.histplot(route_zero, bins=50)
    plt.title("Route-level zero-share distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "route_zero_share.png", dpi=160)
    plt.close()

    print("EDA artifacts saved to:", out_dir)
    print("Rows:", len(train_df), "Routes:", train_df["route_id"].nunique())
    print("Timestamp range:", train_df["timestamp"].min(), "->", train_df["timestamp"].max())


# %%
def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = df[timestamp_col]
    df["hour"] = ts.dt.hour.astype(np.int16)
    df["minute"] = ts.dt.minute.astype(np.int16)
    df["dayofweek"] = ts.dt.dayofweek.astype(np.int16)
    df["day"] = ts.dt.day.astype(np.int16)
    df["weekofyear"] = ts.dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    return df


def add_status_aggregates(df: pd.DataFrame, status_cols: Sequence[str]) -> pd.DataFrame:
    if not status_cols:
        return df

    status_frame = df[list(status_cols)]
    df["status_sum"] = status_frame.sum(axis=1).astype(np.float32)
    df["status_mean"] = status_frame.mean(axis=1).astype(np.float32)
    df["status_std"] = status_frame.std(axis=1).fillna(0).astype(np.float32)
    df["status_max"] = status_frame.max(axis=1).astype(np.float32)
    df["status_nonzero_cnt"] = (status_frame > 0).sum(axis=1).astype(np.float32)

    denom = np.clip(df["status_sum"].to_numpy(dtype=np.float32), 1e-4, None)
    for col in status_cols:
        df[f"{col}_share"] = (df[col].to_numpy(dtype=np.float32) / denom).astype(np.float32)
    return df


def add_group_lags_and_rollings(
    df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int],
    windows: Sequence[int],
) -> pd.DataFrame:
    grouped = df.groupby("route_id", sort=False)
    for col in cols:
        base = grouped[col]
        for lag in lags:
            df[f"{col}_lag_{lag}"] = base.shift(lag).astype(np.float32)
        for window in windows:
            shifted = base.shift(1)
            df[f"{col}_roll_mean_{window}"] = (
                shifted.rolling(window, min_periods=max(2, window // 3)).mean().reset_index(level=0, drop=True).astype(np.float32)
            )
            df[f"{col}_roll_std_{window}"] = (
                shifted.rolling(window, min_periods=max(2, window // 3)).std().reset_index(level=0, drop=True).fillna(0).astype(np.float32)
            )
    return df


def add_delta_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby("route_id", sort=False)
    for col in cols:
        df[f"{col}_diff_1"] = grouped[col].diff(1).astype(np.float32)
        df[f"{col}_diff_2"] = grouped[col].diff(2).astype(np.float32)
    return df


# %%
def build_feature_frame(train_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    status_cols = infer_status_cols(train_df)

    frame = train_df.copy()
    frame = add_future_targets(frame, cfg.target_col, cfg.future_target_cols)
    frame = add_calendar_features(frame)
    frame = add_status_aggregates(frame, status_cols)
    frame = add_group_lags_and_rollings(frame, [cfg.target_col, *status_cols], cfg.lags, cfg.windows)
    frame = add_delta_features(frame, [cfg.target_col, *status_cols])

    grouped = frame.groupby("route_id", sort=False)
    frame["row_number_in_route"] = grouped.cumcount().astype(np.int32)
    frame["route_target_expanding_mean"] = grouped[cfg.target_col].transform(
        lambda s: s.shift(1).expanding().mean()
    ).astype(np.float32)

    supervised = frame.dropna(subset=cfg.future_target_cols).copy()
    supervised = supervised[supervised["row_number_in_route"] >= cfg.min_history].copy()

    excluded = {"id", "timestamp", *cfg.future_target_cols}
    feature_cols = [col for col in supervised.columns if col not in excluded]
    supervised = safe_float32(supervised, [c for c in feature_cols if c != "route_id"])
    return supervised, feature_cols


# %%
def build_long_direct_dataset(
    base_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    frames = []
    for horizon, target_col in enumerate(cfg.future_target_cols, start=1):
        selected_cols = list(dict.fromkeys([*feature_cols, "route_id", "timestamp", target_col]))
        part = base_df[selected_cols].copy()
        part["horizon"] = horizon
        part["y"] = part[target_col]
        part["forecast_timestamp"] = part["timestamp"] + pd.to_timedelta(horizon * cfg.freq_minutes, unit="m")
        part["forecast_hour"] = part["forecast_timestamp"].dt.hour.astype(np.int16)
        part["forecast_dow"] = part["forecast_timestamp"].dt.dayofweek.astype(np.int16)
        part["forecast_hour_sin"] = np.sin(2 * np.pi * part["forecast_hour"] / 24).astype(np.float32)
        part["forecast_hour_cos"] = np.cos(2 * np.pi * part["forecast_hour"] / 24).astype(np.float32)
        part["forecast_dow_sin"] = np.sin(2 * np.pi * part["forecast_dow"] / 7).astype(np.float32)
        part["forecast_dow_cos"] = np.cos(2 * np.pi * part["forecast_dow"] / 7).astype(np.float32)
        frames.append(part.drop(columns=[target_col]))

    long_df = pd.concat(frames, axis=0, ignore_index=True)
    cat_features = ["route_id", "horizon", "forecast_hour", "forecast_dow", "hour", "dayofweek", "is_weekend"]
    all_features = [col for col in long_df.columns if col not in {"timestamp", "forecast_timestamp", "y"}]
    return long_df, all_features, cat_features


# %%
def make_time_splits(base_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_ts = base_df["timestamp"].max()
    valid_start = max_ts - pd.Timedelta(days=cfg.valid_days)
    train_start = max_ts - pd.Timedelta(days=cfg.train_days)

    train_df = base_df[(base_df["timestamp"] >= train_start) & (base_df["timestamp"] < valid_start)].copy()
    valid_df = base_df[base_df["timestamp"] >= valid_start].copy()
    return train_df, valid_df


# %%
def catboost_fit_predict(
    train_long: pd.DataFrame,
    valid_long: pd.DataFrame,
    features: Sequence[str],
    cat_features: Sequence[str],
    cfg: Config,
) -> Tuple[CatBoostRegressor, np.ndarray]:
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cfg.catboost_iterations,
        learning_rate=0.03,
        depth=8,
        random_strength=0.5,
        l2_leaf_reg=8.0,
        subsample=0.85,
        bootstrap_type="Bernoulli",
        grow_policy="SymmetricTree",
        task_type="GPU",
        devices="0",
        random_seed=cfg.random_state,
        verbose=250,
    )

    train_pool = Pool(train_long[features], label=np.log1p(train_long["y"]), cat_features=list(cat_features))
    valid_pool = Pool(valid_long[features], label=np.log1p(valid_long["y"]), cat_features=list(cat_features))
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    valid_pred = np.expm1(model.predict(valid_pool))
    valid_pred = np.clip(valid_pred, 0, None)
    return model, valid_pred


def xgb_fit_predict(
    train_long: pd.DataFrame,
    valid_long: pd.DataFrame,
    features: Sequence[str],
    cat_features: Sequence[str],
    cfg: Config,
) -> Tuple[XGBRegressor, np.ndarray]:
    train_encoded = pd.get_dummies(train_long[features], columns=list(cat_features), dummy_na=True)
    valid_encoded = pd.get_dummies(valid_long[features], columns=list(cat_features), dummy_na=True)
    valid_encoded = valid_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=cfg.xgb_estimators,
        learning_rate=cfg.xgb_learning_rate,
        max_depth=8,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda",
        random_state=cfg.random_state,
    )
    model.fit(
        train_encoded,
        np.log1p(train_long["y"]),
        eval_set=[(valid_encoded, np.log1p(valid_long["y"]))],
        verbose=200,
    )
    valid_pred = np.expm1(model.predict(valid_encoded))
    valid_pred = np.clip(valid_pred, 0, None)
    return model, valid_pred


# %%
def long_to_wide_predictions(pred_df: pd.DataFrame, prediction_col: str, cfg: Config) -> pd.DataFrame:
    wide = pred_df.pivot_table(
        index=["route_id", "timestamp"],
        columns="horizon",
        values=prediction_col,
        aggfunc="first",
    ).reset_index()

    rename_map = {step: f"target_step_{step}" for step in range(1, cfg.forecast_points + 1)}
    wide = wide.rename(columns=rename_map)
    return wide


def reshape_truth(base_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    keep_cols = ["route_id", "timestamp", *cfg.future_target_cols]
    return base_df[keep_cols].copy()


def evaluate_wide_predictions(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: Config,
    label: str,
) -> float:
    merged = truth_df.merge(pred_df, on=["route_id", "timestamp"], suffixes=("_true", "_pred"), how="inner")
    y_true = merged[[f"{c}_true" for c in cfg.future_target_cols]].to_numpy()
    y_pred = merged[[f"{c}_pred" for c in cfg.future_target_cols]].to_numpy()
    score = METRIC.calculate(y_true.reshape(-1), y_pred.reshape(-1))
    print(f"{label}: {score:.5f}")
    return score


# %%
def calibrate_scale(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    best_scale = 1.0
    best_score = METRIC.calculate(y_true, y_pred)
    for scale in np.linspace(0.85, 1.20, 71):
        score = METRIC.calculate(y_true, y_pred * scale)
        if score < best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, best_score


def optimize_two_model_blend(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Tuple[float, float, float]:
    best_weight = 1.0
    best_scale = 1.0
    best_score = METRIC.calculate(y_true, pred_a)
    for weight in np.linspace(0.0, 1.0, 41):
        blend = weight * pred_a + (1.0 - weight) * pred_b
        scale, score = calibrate_scale(y_true, blend)
        if score < best_score:
            best_score = score
            best_weight = float(weight)
            best_scale = float(scale)
    return best_weight, best_scale, best_score


# %%
def make_inference_base(supervised_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    inference_df = supervised_df.sort_values("timestamp").groupby("route_id", sort=False).tail(1).copy()
    selected_cols = list(dict.fromkeys(["route_id", "timestamp", *feature_cols]))
    return inference_df[selected_cols].copy()


def make_long_inference_frame(
    inference_base: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    parts = []
    for horizon in range(1, cfg.forecast_points + 1):
        part = inference_base.copy()
        part["horizon"] = horizon
        part["forecast_timestamp"] = part["timestamp"] + pd.to_timedelta(horizon * cfg.freq_minutes, unit="m")
        part["forecast_hour"] = part["forecast_timestamp"].dt.hour.astype(np.int16)
        part["forecast_dow"] = part["forecast_timestamp"].dt.dayofweek.astype(np.int16)
        part["forecast_hour_sin"] = np.sin(2 * np.pi * part["forecast_hour"] / 24).astype(np.float32)
        part["forecast_hour_cos"] = np.cos(2 * np.pi * part["forecast_hour"] / 24).astype(np.float32)
        part["forecast_dow_sin"] = np.sin(2 * np.pi * part["forecast_dow"] / 7).astype(np.float32)
        part["forecast_dow_cos"] = np.cos(2 * np.pi * part["forecast_dow"] / 7).astype(np.float32)
        parts.append(part)
    return pd.concat(parts, axis=0, ignore_index=True)


# %%
def make_route_histories(
    df: pd.DataFrame,
    target_col: str,
    anchor_timestamps: Optional[Sequence[pd.Timestamp]] = None,
) -> Dict[Tuple[int, pd.Timestamp], np.ndarray]:
    histories: Dict[Tuple[int, pd.Timestamp], np.ndarray] = {}

    if anchor_timestamps is None:
        anchor_timestamps = [df["timestamp"].max()]

    for anchor_ts in anchor_timestamps:
        hist_df = df[df["timestamp"] <= anchor_ts]
        for route_id, route_hist in hist_df.groupby("route_id", sort=False):
            histories[(route_id, pd.Timestamp(anchor_ts))] = route_hist[target_col].to_numpy(dtype=np.float32)
    return histories


def build_foundation_validation_frame(
    supervised_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    anchor_ts = sorted(supervised_df["timestamp"].unique())[-cfg.holdout_anchor_limit :]
    anchor_set = set(anchor_ts)
    valid_anchor_df = supervised_df[supervised_df["timestamp"].isin(anchor_set)].copy()
    return valid_anchor_df


# %%
def predict_timesfm_from_histories(
    history_map: Dict[Tuple[int, pd.Timestamp], np.ndarray],
    cfg: Config,
) -> pd.DataFrame:
    from transformers import TimesFm2_5ModelForPrediction
    import torch

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers",
        device_map="auto",
    )
    route_ids, timestamps, inputs = [], [], []
    for (route_id, anchor_ts), hist in history_map.items():
        if len(hist) < cfg.min_history:
            continue
        route_ids.append(route_id)
        timestamps.append(anchor_ts)
        inputs.append(torch.tensor(hist[-1024:], dtype=torch.float32, device=model.device))

    if not inputs:
        return pd.DataFrame(columns=["route_id", "timestamp", *cfg.future_target_cols])

    with torch.no_grad():
        outputs = model(
            past_values=inputs,
            future_length=cfg.forecast_points,
            return_dict=True,
        )
        preds = outputs.mean_predictions.detach().cpu().numpy()

    pred_df = pd.DataFrame(preds, columns=cfg.future_target_cols)
    pred_df.insert(0, "timestamp", timestamps)
    pred_df.insert(0, "route_id", route_ids)
    for col in cfg.future_target_cols:
        pred_df[col] = np.clip(pred_df[col], 0, None)
    return pred_df


def predict_timer_from_histories(
    history_map: Dict[Tuple[int, pd.Timestamp], np.ndarray],
    cfg: Config,
) -> pd.DataFrame:
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "thuml/timer-base-84m",
        trust_remote_code=True,
        device_map="auto",
    )

    route_ids, timestamps, preds_all = [], [], []
    for (route_id, anchor_ts), hist in history_map.items():
        if len(hist) < cfg.min_history:
            continue

        series = hist[-2048:].astype(np.float32)
        mean = float(series.mean())
        std = float(series.std() + 1e-6)
        normed = (series - mean) / std

        seq = torch.tensor(normed[None, :], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            generated = model.generate(seq, max_new_tokens=cfg.forecast_points)
        pred = generated[:, -cfg.forecast_points :].detach().cpu().numpy()[0]
        pred = np.clip(pred * std + mean, 0, None)

        route_ids.append(route_id)
        timestamps.append(anchor_ts)
        preds_all.append(pred)

    pred_df = pd.DataFrame(preds_all, columns=cfg.future_target_cols)
    pred_df.insert(0, "timestamp", timestamps)
    pred_df.insert(0, "route_id", route_ids)
    return pred_df


# %%
def try_foundation_model(
    model_name: str,
    history_map: Dict[Tuple[int, pd.Timestamp], np.ndarray],
    cfg: Config,
) -> Optional[pd.DataFrame]:
    try:
        if model_name == "timesfm":
            return predict_timesfm_from_histories(history_map, cfg)
        if model_name == "timer":
            return predict_timer_from_histories(history_map, cfg)
    except Exception as exc:
        print(f"{model_name} skipped due to error: {exc}")
    return None


# %%
download_solo_data(CFG)
train_df, test_df = load_solo_data(CFG)
status_cols = infer_status_cols(train_df)
run_large_eda(train_df, status_cols, CFG)


# %%
supervised_df, feature_cols = build_feature_frame(train_df, CFG)
base_train_df, base_valid_df = make_time_splits(supervised_df, CFG)
train_long, long_features, cat_features = build_long_direct_dataset(base_train_df, feature_cols, CFG)
valid_long, _, _ = build_long_direct_dataset(base_valid_df, feature_cols, CFG)

print("Supervised shape:", supervised_df.shape)
print("Train anchors:", base_train_df.shape, "Valid anchors:", base_valid_df.shape)
print("Long train:", train_long.shape, "Long valid:", valid_long.shape)
print("Feature count:", len(long_features))


# %%
cat_model, cat_valid_pred = catboost_fit_predict(train_long, valid_long, long_features, cat_features, CFG)
cat_valid_long = valid_long[["route_id", "timestamp", "horizon"]].copy()
cat_valid_long["prediction"] = cat_valid_pred
cat_valid_wide = long_to_wide_predictions(cat_valid_long, "prediction", CFG)
valid_truth = reshape_truth(base_valid_df, CFG)
cat_score = evaluate_wide_predictions(valid_truth, cat_valid_wide, CFG, "CatBoost wide score")


# %%
xgb_model, xgb_valid_pred = xgb_fit_predict(train_long, valid_long, long_features, cat_features, CFG)
xgb_valid_long = valid_long[["route_id", "timestamp", "horizon"]].copy()
xgb_valid_long["prediction"] = xgb_valid_pred
xgb_valid_wide = long_to_wide_predictions(xgb_valid_long, "prediction", CFG)
xgb_score = evaluate_wide_predictions(valid_truth, xgb_valid_wide, CFG, "XGBoost wide score")


# %%
wide_pred_candidates: Dict[str, pd.DataFrame] = {
    "catboost": cat_valid_wide,
    "xgboost": xgb_valid_wide,
}
wide_scores = {"catboost": cat_score, "xgboost": xgb_score}

foundation_valid = build_foundation_validation_frame(base_valid_df, CFG)
history_map = make_route_histories(train_df, CFG.target_col, sorted(foundation_valid["timestamp"].unique()))

if CFG.run_timesfm:
    timesfm_valid = try_foundation_model("timesfm", history_map, CFG)
    if timesfm_valid is not None and not timesfm_valid.empty:
        timesfm_valid = foundation_valid[["route_id", "timestamp", *CFG.future_target_cols]].merge(
            timesfm_valid,
            on=["route_id", "timestamp"],
            how="inner",
            suffixes=("_true", ""),
        )
        if not timesfm_valid.empty:
            timesfm_pred = timesfm_valid[CFG.future_target_cols].to_numpy()
            timesfm_true = timesfm_valid[[f"{c}_true" for c in CFG.future_target_cols]].to_numpy()
            timesfm_score = METRIC.calculate(timesfm_true.reshape(-1), timesfm_pred.reshape(-1))
            print(f"TimesFM score on anchor holdout: {timesfm_score:.5f}")
            wide_scores["timesfm_anchor"] = timesfm_score

if CFG.run_timer:
    timer_valid = try_foundation_model("timer", history_map, CFG)
    if timer_valid is not None and not timer_valid.empty:
        timer_valid = foundation_valid[["route_id", "timestamp", *CFG.future_target_cols]].merge(
            timer_valid,
            on=["route_id", "timestamp"],
            how="inner",
            suffixes=("_true", ""),
        )
        if not timer_valid.empty:
            timer_pred = timer_valid[CFG.future_target_cols].to_numpy()
            timer_true = timer_valid[[f"{c}_true" for c in CFG.future_target_cols]].to_numpy()
            timer_score = METRIC.calculate(timer_true.reshape(-1), timer_pred.reshape(-1))
            print(f"Timer-family score on anchor holdout: {timer_score:.5f}")
            wide_scores["timer_anchor"] = timer_score

print("Validation score summary:", wide_scores)


# %%
cat_merged = valid_truth.merge(cat_valid_wide, on=["route_id", "timestamp"], suffixes=("_true", "_pred"))
xgb_merged = valid_truth.merge(xgb_valid_wide, on=["route_id", "timestamp"], suffixes=("_true", "_pred"))
y_true_flat = cat_merged[[f"{c}_true" for c in CFG.future_target_cols]].to_numpy().reshape(-1)
cat_flat = cat_merged[[f"{c}_pred" for c in CFG.future_target_cols]].to_numpy().reshape(-1)
xgb_flat = xgb_merged[[f"{c}_pred" for c in CFG.future_target_cols]].to_numpy().reshape(-1)

tab_weight, tab_scale, tab_blend_score = optimize_two_model_blend(y_true_flat, cat_flat, xgb_flat)
print("Best CatBoost/XGBoost blend:", {"catboost_weight": tab_weight, "scale": tab_scale, "score": tab_blend_score})


# %%
inference_base = make_inference_base(supervised_df, feature_cols)
inference_long = make_long_inference_frame(inference_base, feature_cols, CFG)

cat_infer_long = inference_long[["route_id", "timestamp", "horizon"]].copy()
cat_infer_long["prediction"] = np.clip(
    np.expm1(cat_model.predict(Pool(inference_long[long_features], cat_features=list(cat_features)))),
    0,
    None,
)
cat_infer_wide = long_to_wide_predictions(cat_infer_long, "prediction", CFG)

xgb_infer_input = pd.get_dummies(inference_long[long_features], columns=list(cat_features), dummy_na=True)
xgb_train_input = pd.get_dummies(train_long[long_features], columns=list(cat_features), dummy_na=True)
xgb_infer_input = xgb_infer_input.reindex(columns=xgb_train_input.columns, fill_value=0)
xgb_infer_long = inference_long[["route_id", "timestamp", "horizon"]].copy()
xgb_infer_long["prediction"] = np.clip(np.expm1(xgb_model.predict(xgb_infer_input)), 0, None)
xgb_infer_wide = long_to_wide_predictions(xgb_infer_long, "prediction", CFG)

submission_pred = cat_infer_wide.merge(xgb_infer_wide, on=["route_id", "timestamp"], suffixes=("_cat", "_xgb"))
for col in CFG.future_target_cols:
    submission_pred[col] = tab_scale * (
        tab_weight * submission_pred[f"{col}_cat"] + (1.0 - tab_weight) * submission_pred[f"{col}_xgb"]
    )
    submission_pred[col] = np.clip(submission_pred[col], 0, None)


# %%
final_history_map = make_route_histories(train_df, CFG.target_col)
timesfm_final = try_foundation_model("timesfm", final_history_map, CFG) if CFG.run_timesfm else None
timer_final = try_foundation_model("timer", final_history_map, CFG) if CFG.run_timer else None

if timesfm_final is not None and not timesfm_final.empty:
    merged = submission_pred.merge(timesfm_final, on=["route_id", "timestamp"], suffixes=("", "_fm"))
    for col in CFG.future_target_cols:
        merged[col] = 0.85 * merged[col] + 0.15 * merged[f"{col}_fm"]
    submission_pred = merged[["route_id", "timestamp", *CFG.future_target_cols]]

if timer_final is not None and not timer_final.empty:
    merged = submission_pred.merge(timer_final, on=["route_id", "timestamp"], suffixes=("", "_timer"))
    for col in CFG.future_target_cols:
        merged[col] = 0.90 * merged[col] + 0.10 * merged[f"{col}_timer"]
    submission_pred = merged[["route_id", "timestamp", *CFG.future_target_cols]]


# %%
submission_long = submission_pred.melt(
    id_vars=["route_id", "timestamp"],
    value_vars=CFG.future_target_cols,
    var_name="step",
    value_name="y_pred",
)
submission_long["step_num"] = submission_long["step"].str.extract(r"(\d+)").astype(int)
route_last_ts = train_df.groupby("route_id")["timestamp"].max().to_dict()
submission_long["forecast_timestamp"] = submission_long.apply(
    lambda row: route_last_ts[row["route_id"]] + pd.to_timedelta(row["step_num"] * CFG.freq_minutes, unit="m"),
    axis=1,
)

submission_joined = test_df.merge(
    submission_long[["route_id", "forecast_timestamp", "y_pred"]],
    left_on=["route_id", "timestamp"],
    right_on=["route_id", "forecast_timestamp"],
    how="left",
)
fallback = float(train_df[CFG.target_col].median())
submission_joined["y_pred"] = submission_joined["y_pred"].fillna(fallback).clip(lower=0)
submission_path = CFG.artifacts_dir / "submission_solo_advanced.csv"
submission_joined[["id", "y_pred"]].to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print(submission_joined[["id", "route_id", "timestamp", "y_pred"]].head())


# %%
notes = {
    "target_metric": "0.20-ish validation target",
    "best_tabular_blend": {
        "catboost_weight": tab_weight,
        "scale": tab_scale,
        "score": tab_blend_score,
    },
    "standalone_scores": wide_scores,
    "recommendation": (
        "If CatBoost is strongest, keep it as the main model and use TimesFM/Timer only as a low-weight regularizer. "
        "If zero-shot foundation forecasts beat tabular models on the latest anchors, increase their blend weight to 0.20-0.35."
    ),
}
pd.Series(notes, dtype="object").to_json(CFG.artifacts_dir / "run_notes.json", force_ascii=False, indent=2)
print(notes)
