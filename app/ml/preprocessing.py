"""METR-LA CSV loading and leakage-safe chronological preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class TrafficData:
    frame: pd.DataFrame
    timestamps: pd.DatetimeIndex
    sensor_ids: list[str]
    quality: dict[str, dict[str, float]]
    timestamp_column: str
    observation_mask: np.ndarray


def _resolve_column(df: pd.DataFrame, name: str | None) -> str | None:
    if not name: return None
    lookup={str(c).strip().lower():str(c) for c in df.columns}
    found=lookup.get(name.strip().lower())
    if not found: raise ValueError(f"Configured CSV column '{name}' does not exist. Available columns: {list(map(str,df.columns))[:30]}")
    return found


def _parse_timestamps(values: pd.Series) -> pd.Series:
    # format="mixed" accepts common CSVs that mix date-only and full datetime rows.
    return pd.to_datetime(values,errors="coerce",format="mixed")


def _timestamp_column(df: pd.DataFrame, configured: str | None = None) -> str:
    explicit=_resolve_column(df,configured)
    if explicit:
        if _parse_timestamps(df[explicit]).notna().mean()<0.9:
            raise ValueError(f"Configured timestamp column '{explicit}' is not parseable for at least 90% of rows.")
        return explicit
    likely = [c for c in df.columns if str(c).strip().lower() in {"timestamp", "datetime", "date", "time", "unnamed: 0"}]
    candidates = likely + [c for c in df.columns if c not in likely]
    for col in candidates:
        parsed = _parse_timestamps(df[col])
        if len(parsed) and parsed.notna().mean() >= 0.9:
            return str(col)
    raise ValueError("Could not identify a timestamp column. Configure the input as wide format with a parseable timestamp column, or map timestamp/sensor/value columns before loading.")


def load_traffic_csv(path: str | Path, zero_is_missing: bool = True, timestamp_column: str | None = None, sensor_id_column: str | None = None, value_column: str | None = None) -> TrafficData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Traffic CSV not found: {path}")
    headers=pd.read_csv(path,nrows=0).columns
    normalized_headers={str(c).strip().lower().replace(" ","_"):str(c) for c in headers}
    configured_sensor=_resolve_column(pd.DataFrame(columns=headers),sensor_id_column)
    sensor_hint=configured_sensor or next((normalized_headers[n] for n in ("sensor_id","sensor","detector_id","detector","node_id") if n in normalized_headers),None)
    raw = pd.read_csv(path,dtype={sensor_hint:str} if sensor_hint else None)
    if raw.empty or len(raw.columns) < 2:
        raise ValueError("CSV must contain a timestamp plus at least one numeric sensor column.")
    ts_col = _timestamp_column(raw,timestamp_column)
    ts = _parse_timestamps(raw[ts_col])
    sid_col=_resolve_column(raw,sensor_id_column)
    value_col=_resolve_column(raw,value_column)
    normalized={str(c).strip().lower().replace(" ","_"):str(c) for c in raw.columns if c!=ts_col}
    if not sid_col: sid_col=next((normalized[n] for n in ("sensor_id","sensor","detector_id","detector","node_id") if n in normalized),None)
    if not value_col: value_col=next((normalized[n] for n in ("speed","traffic_speed","velocity","value","measurement") if n in normalized),None)
    if bool(sid_col) != bool(value_col):
        raise ValueError("Long-format CSV mapping requires both sensor ID and speed/value columns. Set DATASET_SENSOR_ID_COLUMN and DATASET_VALUE_COLUMN.")
    quality_input={}
    if sid_col and value_col:
        source_speed=pd.to_numeric(raw[value_col],errors="coerce")
        invalid=(source_speed.notna() & ~np.isfinite(source_speed)) | (source_speed<0) | (source_speed>200)
        source_zero=source_speed==0
        for sid,positions in raw.groupby(sid_col,dropna=True).groups.items():
            sid=str(sid); quality_input[sid]={"zero_proportion":float(source_zero.loc[positions].mean()),"invalid_proportion":float(invalid.loc[positions].mean())}
        valid_speed=source_speed.mask(invalid)
        if zero_is_missing: valid_speed=valid_speed.mask(source_zero)
        long=pd.DataFrame({"timestamp":ts,"sensor_id":raw[sid_col].astype("string"),"speed":valid_speed}).dropna(subset=["timestamp","sensor_id"])
        data=long.pivot_table(index="timestamp",columns="sensor_id",values="speed",aggfunc="mean",sort=True)
        data.columns=data.columns.map(str)
        data=data.sort_index()
    else:
        frame=raw.drop(columns=[ts_col]).copy()
        excluded={"id","record_id","sensor_id","sensor","detector_id","detector","node_id","latitude","longitude","lat","lon","location","source","data_source"}
        numeric: dict[str,pd.Series]={}
        for col in frame.columns:
            if str(col).strip().lower() in excluded: continue
            values=pd.to_numeric(frame[col],errors="coerce")
            invalid=(values.notna() & ~np.isfinite(values)) | (values<0) | (values>200)
            zero=values==0
            quality_input[str(col)]={"zero_proportion":float(zero.mean()),"invalid_proportion":float(invalid.mean())}
            values=values.mask(invalid)
            if zero_is_missing: values=values.mask(zero)
            # Numeric observations are sensors in wide format; preserve header IDs exactly.
            if values.notna().any(): numeric[str(col)]=values
        if not numeric:
            raise ValueError("No usable numeric sensor columns found. For long format, configure DATASET_SENSOR_ID_COLUMN and DATASET_VALUE_COLUMN.")
        data=pd.DataFrame(numeric); data.insert(0,"timestamp",ts)
        data=data.dropna(subset=["timestamp"]).sort_values("timestamp",kind="stable")
        data=data.groupby("timestamp",as_index=False,sort=True).mean(numeric_only=True).set_index("timestamp").sort_index()
    if data.empty or not len(data.columns):
        raise ValueError("The CSV has no usable traffic speed observations after schema mapping.")
    sensor_ids = list(data.columns)
    data = data.replace([np.inf, -np.inf], np.nan)
    quality = {s: {"missing_proportion": float(data[s].isna().mean()), **quality_input[s]} for s in sensor_ids}
    # Causal forward-fill only; never interpolate from future measurements.
    observation_mask = data.notna().to_numpy(dtype=bool)
    clean = data.ffill()
    if clean.isna().all(axis=None):
        raise ValueError("CSV contains no valid traffic speed measurements after cleaning.")
    # Constant all-missing sensors cannot be meaningfully modeled; reject rather than fabricate.
    valid = [s for s in sensor_ids if data[s].notna().any()]
    if not valid:
        raise ValueError("No sensor has any valid traffic speed observations.")
    if len(valid) != len(sensor_ids):
        missing=[s for s in sensor_ids if s not in valid]
        raise ValueError(f"Sensor columns have no valid speed observations and cannot be modeled: {missing[:20]}. Remove them explicitly or repair the source CSV.")
    clean = clean[valid]
    return TrafficData(clean.astype("float32"), clean.index, valid, {s: quality[s] for s in valid}, ts_col, observation_mask[:, [sensor_ids.index(s) for s in valid]])


class TrainOnlyScaler:
    def fit(self, values: np.ndarray, mask: np.ndarray | None = None) -> "TrainOnlyScaler":
        x = np.asarray(values, dtype=np.float32)
        valid = np.isfinite(x) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(x))
        selected = x[valid]
        if selected.size == 0:
            raise ValueError("Cannot fit scaler without valid training observations.")
        self.mean_ = float(selected.mean())
        self.std_ = float(selected.std()) or 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std_ + self.mean_

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean_, "std": self.std_}


def chronological_split(values: np.ndarray, train: float = .7, validation: float = .1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    a, b = int(n * train), int(n * (train + validation))
    if a < 1 or b <= a or b >= n:
        raise ValueError("Dataset is too short for chronological train/validation/test splits.")
    return values[:a], values[a:b], values[b:]


def make_windows(values: np.ndarray, input_window: int = 12, horizon: int = 12, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError("Expected values shaped [time, sensors, features].")
    if len(x) < input_window + horizon:
        raise ValueError(f"Need at least {input_window + horizon} observations for requested windows.")
    masks = np.isfinite(x) if mask is None else np.asarray(mask, dtype=bool)
    xs, ys, ms = [], [], []
    for start in range(len(x) - input_window - horizon + 1):
        cut = start + input_window
        xs.append(x[start:cut]); ys.append(x[cut:cut + horizon]); ms.append(masks[cut:cut + horizon])
    return np.stack(xs), np.stack(ys), np.stack(ms)

