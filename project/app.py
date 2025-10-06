# app.py
import os
import json
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Deque, Tuple, Optional, List

import joblib
import xgboost
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# =========================
# Environment / Config
# =========================
SITE_NAME = os.getenv("SITE_NAME", "Room Crowdedness")
PUBLIC_ORIGIN = os.getenv("PUBLIC_ORIGIN", "")  # e.g., https://crowd.example.com

# MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")  # optional
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")  # optional
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "")  # payload: {epoch, co2, temp, humid}
DEFAULT_DEVICE_ID = os.getenv("DEVICE_ID", "")

# Data windowing
HOURS_WINDOW = float(os.getenv("HOURS_WINDOW", "6"))  # keep last X hours in memory
PREDICTION_WINDOW_MINUTES = int(os.getenv("PREDICTION_WINDOW_MINUTES", "15"))  # last N minutes fed to model

# Model
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/occupancy_from_co2_model.h5"))
MODEL_INPUT_TYPE = os.getenv("MODEL_INPUT_TYPE", "features")  # "features" | "sequence"
# If "sequence": we build a [T, F] array over the last PREDICTION_WINDOW_MINUTES, resampled to 30s

MAX_OCCUPANCY = float(os.getenv("MAX_OCCUPANCY", "35"))  # used to normalize headcount predictions

# =========================
# App init
# =========================
app = FastAPI(title=SITE_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_ORIGIN] if PUBLIC_ORIGIN else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=FileResponse)
def dashboard():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return FileResponse(str(INDEX_FILE), media_type="text/html", headers=headers)

@app.get("/health")
def health():
    return {"ok": True, "now": datetime.now(timezone.utc).isoformat()}


@app.get("/api/history")
def api_history(device_id: str = DEFAULT_DEVICE_ID, limit: int = 240):
    dq = store.get(device_id)
    if not dq:
        return {"device_id": device_id, "history": [], "latest": None}

    trim_old(device_id)
    points = []
    limit = max(int(limit), 0)
    window = [] if limit == 0 else list(dq)[-limit:]
    for ts, sample in window:
        points.append(
            {
                "at": ts.isoformat(),
                "co2": json_float(sample.get("co2")),
                "temp": json_float(sample.get("temp")),
                "humid": json_float(sample.get("humid")),
            }
        )

    latest = build_prediction_message(device_id)
    return {"device_id": device_id, "history": points, "latest": latest}

# =========================
# WebSocket "rooms" per device
# =========================
class Rooms:
    def __init__(self):
        self._rooms: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, device: str, ws: WebSocket):
        await ws.accept()
        self._rooms[device].append(ws)

    def disconnect(self, device: str, ws: WebSocket):
        if device in self._rooms and ws in self._rooms[device]:
            self._rooms[device].remove(ws)

    async def broadcast(self, device: str, message: Dict[str, Any]):
        dead = []
        for ws in self._rooms.get(device, []):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(device, ws)

rooms = Rooms()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    device_id = websocket.query_params.get("device_id") or DEFAULT_DEVICE_ID
    await rooms.connect(device_id, websocket)
    try:
        # On connect, send a tiny snapshot
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "device_id": device_id,
            "at": datetime.now(timezone.utc).isoformat()
        }))
        while True:
            # Keep open, ignore incoming for now (could handle pings/filters)
            await websocket.receive_text()
    except WebSocketDisconnect:
        rooms.disconnect(device_id, websocket)

# =========================
# In-memory store
# device_id -> deque[(ts, dict)]
# =========================
store: Dict[str, Deque[Tuple[datetime, Dict[str, Any]]]] = defaultdict(lambda: deque())

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def trim_old(device_id: str):
    cutoff = now_utc() - timedelta(hours=HOURS_WINDOW)
    dq = store[device_id]
    while dq and dq[0][0] < cutoff:
        dq.popleft()

def make_df(device_id: str) -> Optional[pd.DataFrame]:
    trim_old(device_id)
    if not store[device_id]:
        return None
    rows = []
    for ts, payload in list(store[device_id]):
        rows.append({
            "ts": ts,
            "co2": safe_float(payload.get("co2")),
            "temp": safe_float(payload.get("temp")),
            "humid": safe_float(payload.get("humid")),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df

def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def json_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def build_pipeline_features(df: pd.DataFrame, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    if not feature_cols:
        return None
    dfx = df.copy()
    if "co2" not in dfx.columns:
        return None
    co2 = dfx["co2"].dropna()
    if co2.empty:
        return None
    last_idx = co2.index[-1]

    def lag_value(steps: int) -> Optional[float]:
        if len(co2) <= steps:
            return None
        return float(co2.iloc[-(steps + 1)])

    feats: Dict[str, float] = {}
    for col in feature_cols:
        if col == "co2":
            feats[col] = float(co2.iloc[-1])
        elif col.startswith("co2_lag_"):
            try:
                lag = int(col.split("_")[-1])
            except ValueError:
                return None
            val = lag_value(lag)
            if val is None:
                return None
            feats[col] = val
        elif col == "co2_roll_mean_5":
            if len(co2) < 5:
                return None
            feats[col] = float(co2.iloc[-5:].mean())
        elif col == "co2_roll_std_5":
            if len(co2) < 5:
                return None
            feats[col] = float(co2.iloc[-5:].std(ddof=0))
        elif col == "co2_diff_1":
            if len(co2) < 2:
                return None
            feats[col] = float(co2.iloc[-1] - co2.iloc[-2])
        elif col == "co2_pct_change_1":
            if len(co2) < 2 or co2.iloc[-2] == 0:
                return None
            feats[col] = float((co2.iloc[-1] - co2.iloc[-2]) / co2.iloc[-2])
        elif col == "hour":
            feats[col] = float(last_idx.hour)
        elif col == "dayofweek":
            feats[col] = float(last_idx.weekday())
        else:
            # Allow passthrough for additional numeric columns if present
            if col in dfx.columns and dfx[col].notna().any():
                feats[col] = float(dfx[col].dropna().iloc[-1])
            else:
                return None

    return pd.DataFrame([feats])[feature_cols]


# Normalise feature dicts so they can be serialised cleanly
def normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in features.items():
        key_str = str(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            val = float(value)
            if np.isfinite(val):
                clean[key_str] = val
        elif value is not None:
            clean[key_str] = value
    return clean


# =========================
# Model loading (Keras / sklearn)
# =========================
class OccupancyModel:
    def __init__(self, path: Path):
        self.path = path
        self.mode: Optional[str] = None
        self.ok = False
        self.model_label = path.name
        self.keras_model = None
        self.pipeline = None
        self.feature_cols: List[str] = []

        if not path.exists():
            print(f"[Model] Not found at {path} — using heuristic fallback.")
            return

        if self._looks_like_keras(path):
            self._load_keras(path)
        else:
            self._load_pipeline(path)

    # ---------- loaders ----------
    def _load_keras(self, path: Path) -> None:
        try:
            import tensorflow as tf  # type: ignore[import]

            keras = getattr(tf, "keras", None)
            if keras is None:
                raise RuntimeError("TensorFlow installation missing keras module")

            self.keras_model = keras.models.load_model(str(path))
            self.mode = "keras"
            self.ok = True
            print(f"[Model] Loaded Keras model: {path}")
        except Exception as exc:
            print(f"[Model] ERROR loading {path}: {exc}")

    def _load_pipeline(self, path: Path) -> None:
        try:
            artifact = joblib.load(path)
        except ModuleNotFoundError as exc:
            print(f"[Model] Missing dependency while loading {path}: {exc}")
            return
        except Exception as exc:
            print(f"[Model] ERROR loading {path}: {exc}")
            return

        pipeline = None
        feature_cols: List[str] = []
        if isinstance(artifact, dict):
            pipeline = artifact.get("pipeline")
            cols = artifact.get("feature_cols")
            if isinstance(cols, list):
                feature_cols = [str(c) for c in cols]
        else:
            pipeline = artifact

        if pipeline is None:
            print(f"[Model] Joblib artifact {path} missing pipeline entry — fallback.")
            return

        self.pipeline = pipeline
        self.feature_cols = feature_cols
        self.mode = "sklearn"
        self.ok = True
        print(f"[Model] Loaded sklearn pipeline: {path} (features={self.feature_cols or 'auto'})")

    # ---------- predictions ----------
    def predict(self, df: pd.DataFrame) -> Tuple[Optional[float], Dict[str, Any]]:
        if not self.ok or self.mode is None:
            return None, {"model": "heuristic"}

        if self.mode == "keras":
            return self._predict_keras(df)
        if self.mode == "sklearn":
            return self._predict_pipeline(df)
        return None, {"model": "heuristic"}

    def _predict_keras(self, df: pd.DataFrame) -> Tuple[Optional[float], Dict[str, Any]]:
        if self.keras_model is None:
            return None, {"model": self.model_label}

        if MODEL_INPUT_TYPE == "sequence":
            X_seq = sequence_from_df(df)
            if X_seq is None:
                return None, {"model": self.model_label}
            score = self._keras_predict_array(X_seq)
            return score, {"model": self.model_label, "features": {}}

        feats = features_from_df(df)
        if not feats:
            return None, {"model": self.model_label, "features": {}}
        X_cols = sorted(feats.keys())
        X = np.array([[feats[c] for c in X_cols]], dtype=np.float32)
        score = self._keras_predict_array(X)
        return score, {"model": self.model_label, "features": feats}

    def _predict_pipeline(self, df: pd.DataFrame) -> Tuple[Optional[float], Dict[str, Any]]:
        if self.pipeline is None:
            return None, {"model": self.model_label}

        features_df = build_pipeline_features(df, self.feature_cols or [])
        if features_df is None:
            return None, {"model": self.model_label}

        try:
            y = self.pipeline.predict(features_df)
            people_estimate = float(np.squeeze(y))
        except Exception as exc:
            print(f"[Model] pipeline predict error: {exc}")
            return None, {"model": self.model_label, "features": features_df.iloc[0].to_dict()}

        score = None
        if MAX_OCCUPANCY > 0:
            score = float(np.clip(people_estimate / MAX_OCCUPANCY, 0.0, 1.0))

        meta = {
            "model": self.model_label,
            "features": features_df.iloc[0].to_dict(),
            "people_estimate": people_estimate,
        }
        return score, meta

    def _keras_predict_array(self, X: np.ndarray) -> Optional[float]:
        if self.keras_model is None:
            return None
        try:
            y = self.keras_model.predict(X, verbose=0)
            score = float(np.squeeze(y))
            if not (0.0 <= score <= 1.0):
                score = float(1.0 / (1.0 + np.exp(-score)))
            return score
        except Exception as exc:
            print(f"[Model] predict error: {exc}")
            return None

    @staticmethod
    def _header_hex(path: Path) -> str:
        try:
            return path.read_bytes()[:8].hex()
        except Exception:
            return "?"

    @classmethod
    def _looks_like_keras(cls, path: Path) -> bool:
        try:
            header = path.read_bytes()[:8]
        except Exception:
            return False
        return header.startswith(b"\x89HDF")


occupancy_model = OccupancyModel(MODEL_PATH)

# =========================
# Feature builder
# =========================
def features_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build features for "features" input type. Adjust to match your notebook.
    """
    out: Dict[str, Any] = {}
    # Focus on last PREDICTION_WINDOW_MINUTES
    since = now_utc() - timedelta(minutes=PREDICTION_WINDOW_MINUTES)
    dfx = df[df.index >= since]
    if dfx.empty:
        return out

    co2 = dfx["co2"].dropna()
    temp = dfx["temp"].dropna()
    humid = dfx["humid"].dropna()

    if len(co2):
        out["co2_last"] = float(co2.iloc[-1])
        out["co2_rm_5m"] = float(co2.rolling("5min").mean().iloc[-1])
        out["co2_rm_15m"] = float(co2.rolling("15min").mean().iloc[-1])

        # simple ROC via linear fit (ppm/hour)
        t = co2.index.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        if len(t) >= 3 and np.ptp(t) > 0:
            A = np.vstack([t, np.ones_like(t)]).T
            m, b = np.linalg.lstsq(A, co2.to_numpy(), rcond=None)[0]
            out["co2_roc_ppmh"] = float(m * 3600.0)

    if len(temp):
        out["temp_last"] = float(temp.iloc[-1])
    if len(humid):
        out["humid_last"] = float(humid.iloc[-1])

    return out

def sequence_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Build [T, F] sequence over last N minutes, resampled to 30s.
    Features: [co2, temp, humid, co2_rm_5m, co2_rm_15m]
    Returns shape (1, T, F) for Keras.
    """
    since = now_utc() - timedelta(minutes=PREDICTION_WINDOW_MINUTES)
    dfx = df[df.index >= since].copy()
    if dfx.empty:
        return None

    # Resample to 30s, forward-fill
    dfx = dfx.resample("30S").mean().ffill()

    # Rolling means on resampled co2
    dfx["co2_rm_5m"] = dfx["co2"].rolling("5min").mean()
    dfx["co2_rm_15m"] = dfx["co2"].rolling("15min").mean()
    dfx = dfx.dropna().astype(np.float32)

    if len(dfx) < 3:
        return None

    seq = dfx[["co2", "temp", "humid", "co2_rm_5m", "co2_rm_15m"]].values
    return seq[np.newaxis, :, :]  # (1, T, F)

def score_to_level(score: Optional[float]) -> Dict[str, Any]:
    if score is None or not np.isfinite(score):
        return {"label": "Unknown", "color": "#6b7280"}
    if score < 0.25: return {"label": "Low", "color": "#10b981"}
    if score < 0.5:  return {"label": "Moderate", "color": "#f59e0b"}
    if score < 0.75: return {"label": "High", "color": "#ef4444"}
    return {"label": "Very High", "color": "#991b1b"}

# =========================
# Prediction + broadcast
# =========================
import asyncio

async_loop: Optional[asyncio.AbstractEventLoop] = None
mqtt_client: Optional[mqtt.Client] = None
mqtt_thread: Optional[threading.Thread] = None

def build_prediction_message(device_id: str) -> Optional[Dict[str, Any]]:
    df = make_df(device_id)
    if df is None or df.empty:
        return None

    co2_series = df["co2"].dropna() if "co2" in df.columns else pd.Series(dtype=float)
    temp_series = df["temp"].dropna() if "temp" in df.columns else pd.Series(dtype=float)
    humid_series = df["humid"].dropna() if "humid" in df.columns else pd.Series(dtype=float)
    co2_now = float(co2_series.iloc[-1]) if not co2_series.empty else None
    temp_now = float(temp_series.iloc[-1]) if not temp_series.empty else None
    humid_now = float(humid_series.iloc[-1]) if not humid_series.empty else None

    score: Optional[float] = None
    model_name = "heuristic"
    people_estimate: Optional[float] = None
    features_dbg: Dict[str, Any] = {}

    if occupancy_model.ok:
        score_candidate, meta = occupancy_model.predict(df)
        meta = meta or {}
        meta_features = meta.get("features") or {}
        if isinstance(meta_features, dict):
            features_dbg = normalize_features(meta_features)
        model_name = str(meta.get("model", occupancy_model.model_label))
        people_val = meta.get("people_estimate")
        if isinstance(people_val, (int, float, np.floating, np.integer)):
            people_estimate = float(people_val)
        if score_candidate is not None:
            score = float(score_candidate)

    if score is None:
        heuristic_feats = features_from_df(df)
        if heuristic_feats:
            features_dbg = normalize_features(heuristic_feats)
        if co2_now is not None and np.isfinite(co2_now):
            score = float(np.clip((co2_now - 400.0) / 1600.0, 0.0, 1.0))

    prediction_block: Dict[str, Any] = {
        "crowded_score": None if score is None else float(score),
        "crowded_level": score_to_level(score),
        "features": features_dbg,
        "model": model_name,
    }
    if people_estimate is not None and np.isfinite(people_estimate):
        prediction_block["people_estimate"] = float(people_estimate)

    return {
        "type": "prediction",
        "device_id": device_id,
        "at": now_utc().isoformat(),
        "co2": co2_now,
        "temp": temp_now,
        "humid": humid_now,
        "prediction": prediction_block,
    }


def on_new_sample(device_id: str):
    """Process a freshly arrived telemetry sample."""
    msg = build_prediction_message(device_id)
    if msg is None:
        return

    # Broadcast on the FastAPI loop
    loop = async_loop
    if not loop or not loop.is_running():
        print("[MQTT] WebSocket loop not ready; skipping broadcast")
        return
    asyncio.run_coroutine_threadsafe(rooms.broadcast(device_id, msg), loop)

# =========================
# MQTT consumption
# =========================
def parse_payload(payload: bytes) -> Optional[Dict[str, Any]]:
    """
    Expect JSON: {"epoch": 1700000000, "co2": 812.0, "temp": 27.1, "humid": 55.2}
    """
    try:
        obj = json.loads(payload.decode("utf-8"))
        return obj
    except Exception:
        return None

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected rc={rc}")
    client.subscribe(MQTT_TOPIC, qos=1)
    print(f"[MQTT] Subscribed to {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    device_id = DEFAULT_DEVICE_ID  # extend here if you want device from topic
    obj = parse_payload(msg.payload)
    if not obj:
        return
    # epoch (s or ms)
    epoch = obj.get("timestamp")
    if epoch is None:  # skip if no time
        return

    # Normalize epoch to seconds -> datetime with tz
    if epoch > 1e12:  # ms
        epoch = epoch / 1000.0
    ts = datetime.fromtimestamp(float(epoch), tz=timezone.utc)

    sample = {
        "co2": safe_float(obj.get("co2")),
        "temp": safe_float(obj.get("temperature")),
        "humid": safe_float(obj.get("humidity")),
    }
    store[device_id].append((ts, sample))
    print(f"[MQTT] {device_id} @ {ts.isoformat()} co2={sample['co2']} temp={sample['temp']} humid={sample['humid']}")
    trim_old(device_id)
    on_new_sample(device_id)

def start_mqtt():
    global mqtt_client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client = client
    try:
        client.loop_forever()
    finally:
        mqtt_client = None


def ensure_mqtt_thread():
    global mqtt_thread
    if mqtt_thread and mqtt_thread.is_alive():
        return
    mqtt_thread = threading.Thread(target=start_mqtt, name="mqtt", daemon=True)
    mqtt_thread.start()
    print("[MQTT] thread started")


@app.on_event("startup")
async def on_startup():
    global async_loop
    async_loop = asyncio.get_running_loop()
    ensure_mqtt_thread()


@app.on_event("shutdown")
async def on_shutdown():
    global async_loop
    async_loop = None
    if mqtt_client:
        try:
            mqtt_client.disconnect()
        except Exception as exc:
            print(f"[MQTT] disconnect error: {exc}")

# ========== END ==========
