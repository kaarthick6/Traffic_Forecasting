# DCRNN Traffic Flow and Travel-Time Estimation

Python application for METR-LA speed forecasting with a PyTorch DCRNN, FastAPI, Streamlit, PostgreSQL, and Alembic. The application keeps speed forecasting separate from physical route-time estimation: a correlation graph supports DCRNN message passing, while an actual route duration requires supplied road edges with defensible distances.

## Current dataset inspection

The configured local CSV at `C:\Users\Karthick\Downloads\METR-LA (1).csv` is accessible in the development environment. It is wide-format, with `Unnamed: 0` timestamps and 207 numeric sensor-ID columns (for example `773869`); the inspected file contains 34,272 observations at five-minute intervals. IDs are preserved as strings. Repeated timestamps are averaged, records are sorted, and zero speed can be treated as a missing reading (`ZERO_IS_MISSING=true` by default). Inputs use causal forward-fill; leading gaps use the training-only mean after scaling, and original unobserved targets are masked during training and evaluation. No future-reading interpolation is used. Missing sensors are not assigned invented values. The CSV itself contains no coordinates, road topology, or edge distances. A separate local `graph_sensor_locations.csv` contains matching coordinates for all 207 CSV sensors; it has no road edges or distances.

The five-page project report describes LSTM/DL4J with Spring Boot, Android, PostgreSQL, and Kafka. This implementation follows the updated Python stack and replaces LSTM forecasting with DCRNN. The report's LSTM metrics and illustrative 1.47-minute trip example are not represented as DCRNN results; coordinates alone do not establish the driving route or segment distance needed to reproduce a physical trip estimate.

## What works and what requires metadata

- DCRNN speed forecasting uses bidirectional diffusion convolution in recurrent encoder and autoregressive decoder cells. It predicts all graph nodes and only filters the response to requested sensors.
- A validated user-supplied road graph is preferred for DCRNN supports. Without it, the forecasting graph is computed from training-partition speed correlations and stored with a stable sensor-ID mapping. A correlation graph is explicitly **not** a road graph.
- The route service accepts validated directed road edges and positive `edge_distance_km`; optionally, fastest-route selection requires `edge_travel_time_seconds` for every edge. It maps each segment to the downstream sensor speed and converts METR-LA mph into km/h before distance-over-speed calculation.
- With the CSV alone, a driving route and physical duration cannot be calculated. The API reports this instead of fabricating route, distance, or minutes. Supply metadata in the format below to enable those functions.
- PostgreSQL persistence is configured for observations, predictions, routes, valid travel-time estimates, sensors, model metadata, and training jobs. Prediction persistence is best effort while observation ingestion requires a working database.

The CSV loader detects wide data and common long-format column names (`timestamp`, `sensor_id`, `speed`, etc.). For other names, configure `DATASET_TIMESTAMP_COLUMN`, `DATASET_SENSOR_ID_COLUMN`, and `DATASET_VALUE_COLUMN` in `.env`. Set the sensor and value mapping together for long-format data. Sensor identifiers are read as strings so leading zeroes are retained.

### Optional road metadata

Set `SENSOR_METADATA_PATH` to a JSON file shaped like:

```json
{
  "sensors": [
    {"sensor_id": "<ID from the CSV>", "latitude": 34.0, "longitude": -118.0}
  ],
  "edges": [
    {"source_sensor": "<ID from the CSV>", "destination_sensor": "<ID from the CSV>", "edge_distance_km": 1.2, "edge_travel_time_seconds": 90}
  ]
}
```

The example values are schema examples only. Replace them with verified metadata. Every referenced sensor must be present in the CSV; edge lengths must be positive kilometres. A directed edge is one road segment and its downstream sensor supplies the predicted segment speed. Coordinates can support mapping, but do not alone establish road adjacency or segment distances. CSV column order and traffic correlation are never interpreted as road order.

## Windows setup (PowerShell)

Python 3.11 is recommended. PyTorch CPU works; CUDA is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` if needed. `DATASET_PATH` defaults to the supplied local CSV. For another machine, set it to that machine's CSV. For PostgreSQL, install/start PostgreSQL, then create the development database:

```powershell
psql -U postgres -c "CREATE DATABASE traffic_prediction;"
```

Change DB environment values for your local account. The example password is for local development only. Do not commit `.env`.
The API binds to loopback in Compose. To protect training, uploads, and observation ingestion even on a trusted local network, set a nonempty `ADMIN_API_TOKEN` in `.env`; Streamlit sends it to FastAPI automatically. Do not expose the development service directly to an untrusted network.

Run migrations, train the model, and view evaluation metrics:

```powershell
alembic upgrade head
python -m app.ml.train
python -m app.ml.evaluate
```

Training runs on the host when invoked this way. The best validation checkpoint and preprocessing/graph metadata are saved under `artifacts/`; generated model files are ignored by Git. Training also evaluates the held-out chronological test set and writes `metrics.json`, `training_history.json`, and the best `dcrnn.pt` checkpoint. Metrics are emitted only after an actual run.

Start the API and UI in separate terminals:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
python -m streamlit run streamlit_app/app.py --server.port 8501
```

Open [Streamlit](http://localhost:8501) or [FastAPI docs](http://127.0.0.1:8000/docs). The API auto-loads the configured CSV at startup. The dashboard also supports CSV upload.

Useful PowerShell checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/dataset/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/graph/status
python -m pytest
```

## Docker Compose

Training is a host command by default. First copy/edit `.env`; set `DATASET_HOST_FILE` and `SENSOR_METADATA_HOST_FILE` to the full Windows paths for the CSV and coordinate metadata. Use forward-slash paths, for example `C:/Users/Karthick/Downloads/METR-LA (1).csv`. Compose mounts those files read-only inside the API container; model artifacts are mounted at `./artifacts`. If you do not have coordinate metadata, remove the metadata bind mount and set `SENSOR_METADATA_PATH` to empty in the `api` service. Then run:

```powershell
docker compose up --build
```

PostgreSQL starts first with a health check and the API runs Alembic migrations. The API and Streamlit communicate over the `api` service name. Compose publishes PostgreSQL 5432, API 8000, and UI 8501. If training on the host, it needs its own Python environment with CPU PyTorch and the same artifact directory as the Compose mount.

## Configuration and model details

All model defaults can be set in `.env`: 12 history steps, 12 forecast steps (five-minute intervals), diffusion order 2, 64 hidden channels, two encoder/decoder layers, batch 32, Adam at 0.001, up to 100 epochs, and patience 10. Data splits are chronological 70/10/20. The scaler and correlation graph use training data only. Random seeds are fixed at 42. CPU is selected unless CUDA is available. Scheduled sampling decays linearly from 0.5 to zero. Training and validation losses are masked MAE in normalized space; held-out MAE/MSE/RMSE are reported in mph overall and by horizon.

Artifacts include checkpoint, scaler values, node mapping, adjacency, graph source/configuration, model settings, training history, and test metrics. Missing checkpoint means inference readiness is false; no random weights are served.

## API overview

- `GET /health`, `GET /ready`, `GET /api/v1/system/status`
- `POST /api/v1/dataset/load`, `POST /api/v1/dataset/upload`, `GET /api/v1/dataset/status`, `/preview`, `/statistics`
- `GET /api/v1/sensors`, `GET /api/v1/sensors/{sensor_id}`
- `GET /api/v1/graph/status`, `GET /api/v1/graph/route?source_id=...&destination_id=...`
- `POST /api/v1/predictions` with `{"sensor_ids":["..."],"horizon":1}`
- `POST /api/v1/travel-time/estimate` with source/destination IDs and forecast horizon
- `POST /api/v1/traffic/observations` for validated batched observations
- Optional Kafka consumer status at `GET /api/v1/ingestion/status`; REST ingestion is always available
- `POST /api/v1/model/train`, `GET /api/v1/model/train/{job_id}`, model metrics and metadata endpoints

Inference errors when artifacts are absent, sensor IDs are unknown, history is insufficient, a requested horizon exceeds training, or route metadata is missing. Partial sensor requests still run the complete graph and return selected nodes only.

### Optional Kafka streaming

Set `KAFKA_ENABLED=true`, then start Compose with `docker compose --profile streaming up --build`. The optional single-node Kafka broker exposes `localhost:9094` for host producer tooling and `kafka:9092` to containers; the API consumer retries broker connections and validates each JSON observation before an idempotent PostgreSQL insert. The default is disabled, so Kafka is not needed for CSV or REST workflows. To publish a synthetic demonstration record for a real dataset ID, run `python scripts/publish_demo.py --sensor-id <actual-id> --bootstrap-servers localhost:9094`. The script labels messages `synthetic-demo`; they are simulated and not live traffic.

## Tests and limitations

Unit tests use tiny synthetic fixtures and do not require the METR-LA file for their logic. The suite covers wide/long CSV handling, train-only scaling, windows, graph supports, DCRNN shape/loss/checkpoint behavior, road-path calculation, coordinate validation, unit conversion, streaming inference-window updates, and refusal to calculate time without distances. SQLAlchemy persistence models are exercised with an in-memory SQLite test; PostgreSQL migrations and `ON CONFLICT` ingestion require a live PostgreSQL check. This checkout was created from an empty directory, so there was no prior Android/web client, Java backend, or repository convention to preserve.

