from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class Sensor(Base):
    __tablename__="sensor"
    sensor_id: Mapped[str]=mapped_column(String(80),primary_key=True)
    latitude: Mapped[float|None]=mapped_column(Float)
    longitude: Mapped[float|None]=mapped_column(Float)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())

class TrafficObservation(Base):
    __tablename__="traffic_observation"
    __table_args__=(UniqueConstraint("sensor_id","observed_at",name="uq_observation_sensor_time"),)
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    sensor_id: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"),index=True)
    observed_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),index=True)
    speed: Mapped[float]=mapped_column(Float)
    source: Mapped[str]=mapped_column(String(80),default="api")
    ingested_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())

class Prediction(Base):
    __tablename__="prediction"
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    sensor_id: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"),index=True)
    generated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
    forecast_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),index=True)
    predicted_speed: Mapped[float]=mapped_column(Float)
    horizon_step: Mapped[int]=mapped_column(Integer)
    model_version: Mapped[str]=mapped_column(String(80))

class RouteRecord(Base):
    __tablename__="route"
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    source_sensor: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"),index=True)
    destination_sensor: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"),index=True)
    sensor_sequence: Mapped[list]=mapped_column(JSON)
    graph_version: Mapped[str]=mapped_column(String(120))
    distance_km: Mapped[float|None]=mapped_column(Float)

class TravelTimeEstimate(Base):
    __tablename__="travel_time_estimate"
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    source_sensor: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"))
    destination_sensor: Mapped[str]=mapped_column(ForeignKey("sensor.sensor_id"))
    route: Mapped[list]=mapped_column(JSON)
    distance_km: Mapped[float]=mapped_column(Float)
    estimated_seconds: Mapped[float]=mapped_column(Float)
    model_version: Mapped[str]=mapped_column(String(80))
    estimated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())

class ModelMetadata(Base):
    __tablename__="model_metadata"
    model_version: Mapped[str]=mapped_column(String(80),primary_key=True)
    architecture: Mapped[str]=mapped_column(String(40),default="DCRNN")
    config: Mapped[dict]=mapped_column(JSON)
    metrics: Mapped[dict|None]=mapped_column(JSON)
    artifact_location: Mapped[str]=mapped_column(String(500))
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())

class TrainingJob(Base):
    __tablename__="training_job"
    job_id: Mapped[str]=mapped_column(String(36),primary_key=True)
    status: Mapped[str]=mapped_column(String(30),index=True)
    progress: Mapped[float]=mapped_column(Float,default=0)
    error: Mapped[str|None]=mapped_column(String(2000))
    updated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),onupdate=func.now(),server_default=func.now())

