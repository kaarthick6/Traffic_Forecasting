DCRNN Traffic Flow and Travel-Time Estimation

Python application for METR-LA traffic forecasting with a PyTorch DCRNN, Apache Kafka, FastAPI, Streamlit, PostgreSQL, and Alembic. The system uses an event-driven architecture in which historical METR-LA sensor observations are replayed through Kafka to simulate a live traffic sensor network. Real-time traffic forecasts are generated continuously from streaming observations and served through FastAPI and Streamlit.

The application separates speed forecasting from physical route-time estimation. A correlation graph supports DCRNN message passing, while actual travel-time estimation requires validated road-segment metadata containing physical distances.

System Architecture
                 METR-LA CSV Dataset
                          |
                          v
                Kafka Replay Producer
                          |
                          v
                  Apache Kafka
                          |
          --------------------------------
          |              |              |
          v              v              v
    Observation      PostgreSQL     Monitoring
      Consumer         Storage        Services
          |
          v
    Sliding Window
          |
          v
         DCRNN
    Forecast Engine
          |
          v
       Kafka Topic
          |
          v
        FastAPI
          |
          v
       Streamlit


The historical dataset is continuously replayed through Kafka, allowing the platform to behave like a real-time traffic management system while using recorded METR-LA measurements.

Current Dataset Inspection

The configured local CSV at:

C:\Users\Karthick\Downloads\METR-LA (1).csv


is accessible in the development environment.

The file contains:

34,272 observations
207 traffic sensors
Five-minute intervals
Wide-format structure
Timestamp column stored as Unnamed: 0
Sensor IDs such as 773869, stored as strings

During ingestion:

Repeated timestamps are averaged.
Records are sorted chronologically.
Zero speed can be treated as missing (ZERO_IS_MISSING=true).
Missing values are handled using causal forward filling.
Training-only means are used for leading missing values.
Original missing targets remain masked during training and evaluation.
No future data leakage or interpolation is performed.

The CSV contains speed measurements only.

The dataset does not contain:

Road topology
Segment lengths
Route information
Road adjacency
Travel distances

A separate file:

graph_sensor_locations.csv


contains coordinates for all 207 sensors but does not define road connections or segment distances.

Kafka-Based Real-Time Simulation

The application treats the historical dataset as a stream of traffic sensor events.

Instead of reading observations directly from a CSV during inference, the system replays observations through Kafka in chronological order.

Example:

Historical Dataset

08:00
08:05
08:10
08:15


can be replayed as:

Simulation

t = 0 sec
t = 1 sec
t = 2 sec
t = 3 sec


allowing continuous traffic updates.

Each Kafka message represents a single sensor observation:

{
  "timestamp": "2012-03-01T08:00:00",
  "sensor_id": "773869",
  "speed": 61.3
}


This enables realistic testing of streaming traffic analytics without requiring live roadway sensors.

Kafka Topics
traffic.sensor.raw

Raw observations generated from CSV replay.

Example:

{
  "timestamp": "2012-03-01T08:00:00",
  "sensor_id": "773869",
  "speed": 61.3
}

traffic.sensor.processed

Validated and cleaned observations used for forecasting.

Example:

{
  "timestamp": "2012-03-01T08:00:00",
  "sensor_id": "773869",
  "speed": 61.3
}

traffic.forecasts

DCRNN prediction results.

Example:

{
  "prediction_time": "2012-03-01T08:00:00",
  "sensor_id": "773869",
  "forecast": [
    60.8,
    60.1,
    59.5,
    58.9
  ]
}

traffic.traveltime

Estimated route durations.

Example:

{
  "source_sensor": "773869",
  "destination_sensor": "767541",
  "estimated_minutes": 14.6
}

DCRNN Forecasting Pipeline

The DCRNN forecasting service consumes streaming observations from Kafka.

Workflow:

Kafka Observation
       |
       v
Sliding Window Buffer
       |
       v
 DCRNN Forecast
       |
       v
Kafka Forecast Topic
       |
       v
 PostgreSQL
       |
       v
FastAPI / Streamlit


For every sensor:

buffer[sensor_id].append(speed)


After sufficient history exists:

if len(window) == 12:
    prediction = model.predict(window)


Predictions are published back to Kafka and stored in PostgreSQL.

What Works and What Requires Metadata
Supported
DCRNN forecasting across all graph nodes
Kafka-based real-time simulation
Streaming inference
PostgreSQL persistence
FastAPI APIs
Streamlit visualization
Correlation-based graph construction
Requires Road Metadata

Physical travel-time estimation requires:

Road connectivity
Segment distances
Optional segment travel times

The forecasting graph used by DCRNN is not interpreted as a road network.

If route metadata is absent the API explicitly refuses travel-time calculation instead of fabricating durations.

Optional Road Metadata

Set:

SENSOR_METADATA_PATH


to a JSON file:

{
  "sensors": [
    {
      "sensor_id": "773869",
      "latitude": 34.0,
      "longitude": -118.0
    }
  ],
  "edges": [
    {
      "source_sensor": "773869",
      "destination_sensor": "767541",
      "edge_distance_km": 1.2,
      "edge_travel_time_seconds": 90
    }
  ]
}


Every referenced sensor must be present in the dataset.

Windows Setup (PowerShell)

Python 3.11 recommended.

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env


Install and start:

PostgreSQL
Apache Kafka

Create database:

psql -U postgres -c "CREATE DATABASE traffic_prediction;"

Model Training

Apply migrations and train:

alembic upgrade head

python -m app.ml.train

python -m app.ml.evaluate


Training produces:

artifacts/
├── dcrnn.pt
├── adjacency.npy
├── node_mapping.json
├── scaler.json
├── graph_metadata.json
├── metrics.json
└── training_history.json

Running Kafka Replay Simulation

Start Kafka services first.

Then launch the replay producer:

python scripts/csv_kafka_producer.py


The producer:

Loads METR-LA CSV
Replays observations chronologically
Publishes sensor events to Kafka
Simulates real-time traffic flow
Running Services

Start the API:

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


Start Streamlit:

python -m streamlit run streamlit_app/app.py

Docker Compose

Kafka is a required infrastructure service.

docker compose up --build


The stack includes:

PostgreSQL
Kafka
Zookeeper
FastAPI
Streamlit


Startup order:

PostgreSQL
      ↓
Kafka
      ↓
FastAPI
      ↓
Streamlit

API Overview
System
GET /health
GET /ready
GET /api/v1/system/status

Dataset
POST /api/v1/dataset/load
POST /api/v1/dataset/upload
GET /api/v1/dataset/status
GET /api/v1/dataset/preview
GET /api/v1/dataset/statistics

Sensors
GET /api/v1/sensors
GET /api/v1/sensors/{sensor_id}

Graph
GET /api/v1/graph/status
GET /api/v1/graph/route

Predictions
POST /api/v1/predictions

Travel Time
POST /api/v1/travel-time/estimate

Traffic Observations
POST /api/v1/traffic/observations

Simulation
POST /api/v1/simulation/start
POST /api/v1/simulation/stop
GET /api/v1/simulation/status

Model Management
POST /api/v1/model/train
GET /api/v1/model/train/{job_id}
GET /api/v1/model/metrics
GET /api/v1/model/metadata

Persistence

PostgreSQL stores:

Sensor observations
Kafka ingestion records
Forecasts
Training jobs
Travel-time calculations
Route definitions
Sensor metadata

Prediction persistence is best effort while observation ingestion requires a healthy database connection.

Tests

The test suite covers:

Kafka producers and consumers
Real-time replay simulation
Wide and long CSV ingestion
Missing-value handling
Scaling and preprocessing
DCRNN training and inference
Forecast persistence
Route calculations
Travel-time estimation
Streaming window updates
API endpoints
PostgreSQL integration
Project Highlights
PyTorch DCRNN for graph-based traffic forecasting
Apache Kafka real-time simulation pipeline
Event-driven microservice architecture
PostgreSQL persistence layer
FastAPI REST endpoints
Streamlit analytics dashboard
Chronological METR-LA replay engine
Real-time forecasting workflow
Route travel-time estimation
Production-style intelligent transportation system design

This architecture demonstrates how historical traffic sensor data can be transformed into a real-time streaming environment where Kafka continuously feeds a DCRNN forecasting engine, enabling realistic traffic monitoring, prediction, and analytics.
