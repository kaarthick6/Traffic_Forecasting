import pytest
pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.db.session import Base
from app.db.models import Sensor, TrafficObservation, Prediction
from datetime import datetime, timezone

def test_sensor_observation_and_prediction_persistence():
    engine=create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(Sensor(sensor_id="sensor-1",latitude=34.0,longitude=-118.0))
        session.flush()
        session.add(TrafficObservation(sensor_id="sensor-1",observed_at=datetime(2024,1,1),speed=42.0,source="test"))
        session.add(Prediction(sensor_id="sensor-1",forecast_at=datetime(2024,1,1,0,5),predicted_speed=40.0,horizon_step=1,model_version="test-dcrnn"))
    with Session(engine) as session:
        assert len(session.scalars(select(TrafficObservation)).all())==1
        assert len(session.scalars(select(Prediction)).all())==1

