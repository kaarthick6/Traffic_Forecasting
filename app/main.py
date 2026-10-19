from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4
import asyncio
import json
import secrets
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, Header, Depends
from pydantic import BaseModel, Field, field_validator
from app.core.config import get_settings
from app.services.dataset_service import DatasetService
from app.services.route_service import RouteService, estimate_route_time
from app.ml.dcrnn import DCRNN
from app.ml.graph import diffusion_supports

cfg=get_settings(); dataset=DatasetService(); route_service: RouteService|None=None
model=None; model_supports=None; artifact_meta=None; scaler=None; trained_at=None
jobs: dict[str,dict]={}; training_lock=Lock(); app=FastAPI(title="DCRNN Traffic Flow and Travel-Time Estimation",version="1.0.0")
kafka_consumer=None; kafka_task=None
kafka_stats={"enabled":False,"connected":False,"accepted":0,"rejected":0,"last_error":None,"topic":cfg.kafka_topic}
observation_sources:dict[tuple[str,str],str]={}
observation_ingestion_stats={"accepted":0,"transport_counts":{},"source_counts":{},"last_observed_at":None}

def source_key(sensor_id:str,observed_at):
    return str(sensor_id),observed_at.isoformat() if hasattr(observed_at,"isoformat") else str(observed_at)

def record_ingestion(rows,transport="rest"):
    for row in rows:
        sid=str(row["sensor_id"]); observed=row["observed_at"]
        if isinstance(observed,str): observed=datetime.fromisoformat(observed)
        source=str(row.get("source") or "api")
        observation_sources[source_key(sid,observed)]=source
        observation_ingestion_stats["source_counts"][source]=observation_ingestion_stats["source_counts"].get(source,0)+1
        observation_ingestion_stats["transport_counts"][transport]=observation_ingestion_stats["transport_counts"].get(transport,0)+1
        observation_ingestion_stats["last_observed_at"]=observed.isoformat()
        observation_ingestion_stats["accepted"]+=1

def require_admin(x_admin_token:str|None=Header(default=None)):
    if cfg.admin_api_token and not secrets.compare_digest(x_admin_token or "",cfg.admin_api_token):
        raise HTTPException(401,"A valid X-Admin-Token header is required for this operation.")

class PredictionRequest(BaseModel):
    sensor_ids: list[str] | None = None
    horizon: int = Field(default=1,ge=1,le=12)

class TravelTimeRequest(BaseModel):
    source_sensor_id: str
    destination_sensor_id: str
    departure_time: datetime | None = None
    forecast_horizon: int = Field(default=1,ge=1,le=12)
    route_strategy: str = "shortest_distance"

class ObservationItem(BaseModel):
    sensor_id: str = Field(min_length=1,max_length=80)
    observed_at: datetime
    speed: float = Field(ge=0,le=200)
    source: str = Field(default="api",max_length=80)

    @field_validator("observed_at")
    @classmethod
    def naive_dataset_time(cls,value):
        if value.tzinfo is not None: raise ValueError("observed_at must match the timezone-naive dataset convention.")
        return value

class ObservationBatch(BaseModel):
    observations: list[ObservationItem] = Field(min_length=1,max_length=5000)

class TrainingRequest(BaseModel):
    input_window: int | None = Field(default=None,ge=1,le=288)
    forecast_horizon: int | None = Field(default=None,ge=1,le=12)
    diffusion_order: int | None = Field(default=None,ge=1,le=5)
    hidden_channels: int | None = Field(default=None,ge=4,le=256)
    encoder_layers: int | None = Field(default=None,ge=1,le=4)
    decoder_layers: int | None = Field(default=None,ge=1,le=4)
    batch_size: int | None = Field(default=None,ge=1,le=256)
    max_epochs: int | None = Field(default=None,ge=1,le=1000)
    early_stopping_patience: int | None = Field(default=None,ge=1,le=100)

def persist_predictions(rows: list[dict]) -> None:
    """Best-effort DB persistence; database status remains independently visible."""
    try:
        from app.db.session import SessionLocal
        from app.db.models import Sensor, Prediction
        with SessionLocal.begin() as session:
            for sensor_id in {row["sensor_id"] for row in rows}:
                if session.get(Sensor,sensor_id) is None: session.add(Sensor(sensor_id=sensor_id))
            session.flush()
            session.add_all([Prediction(sensor_id=r["sensor_id"],forecast_at=datetime.fromisoformat(r["forecast_time"]),predicted_speed=r["predicted_speed"],horizon_step=r["horizon_step"],model_version="dcrnn-best") for r in rows])
    except Exception as exc:
        print(f"Prediction persistence unavailable ({type(exc).__name__}).")

def persist_job(job_id:str,status:str,progress:float,error:str|None=None,metrics:dict|None=None):
    try:
        from app.db.session import SessionLocal
        from app.db.models import TrainingJob,ModelMetadata
        with SessionLocal.begin() as session:
            session.merge(TrainingJob(job_id=job_id,status=status,progress=progress,error=error))
            if metrics is not None and artifact_meta:
                config={k:v for k,v in artifact_meta.items() if k!="adjacency"}
                session.merge(ModelMetadata(model_version="dcrnn-best",architecture="DCRNN",config=config,metrics=metrics,artifact_location=str(cfg.artifact_dir/"dcrnn.pt")))
    except Exception as exc: print(f"Training metadata persistence unavailable ({type(exc).__name__}).")

def sync_sensors():
    if dataset.data is None: return
    try:
        from sqlalchemy.dialects.postgresql import insert
        from app.db.session import SessionLocal
        from app.db.models import Sensor
        rows=[]
        for sid in dataset.data.sensor_ids:
            coords=route_service.node_attributes.get(sid) if route_service else None
            rows.append({"sensor_id":sid,"latitude":coords.get("latitude") if coords else None,"longitude":coords.get("longitude") if coords else None})
        statement=insert(Sensor).values(rows)
        statement=statement.on_conflict_do_update(index_elements=[Sensor.sensor_id],set_={"latitude":statement.excluded.latitude,"longitude":statement.excluded.longitude})
        with SessionLocal.begin() as session: session.execute(statement)
    except Exception as exc: print(f"Sensor registry persistence unavailable ({type(exc).__name__}).")

def persist_model_metadata():
    if not artifact_meta: return
    try:
        from app.db.session import SessionLocal
        from app.db.models import ModelMetadata
        metrics_path=cfg.artifact_dir/"metrics.json"
        metrics=json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
        config={k:v for k,v in artifact_meta.items() if k!="adjacency"}
        with SessionLocal.begin() as session:
            session.merge(ModelMetadata(model_version="dcrnn-best",architecture="DCRNN",config=config,metrics=metrics,artifact_location=str(cfg.artifact_dir/"dcrnn.pt")))
    except Exception as exc: print(f"Model metadata persistence unavailable ({type(exc).__name__}).")

def load_model():
    global model,model_supports,artifact_meta,scaler,trained_at
    meta_path=cfg.artifact_dir/"metadata.json"; ckpt_path=cfg.artifact_dir/"dcrnn.pt"
    model=None; model_supports=None; artifact_meta=None; scaler=None; trained_at=None
    if not meta_path.is_file() or not ckpt_path.is_file(): return False
    try:
        artifact_meta=json.loads(meta_path.read_text(encoding="utf-8")); scaler=artifact_meta["scaler"]
        a=np.asarray(artifact_meta["adjacency"],np.float32); f,b=diffusion_supports(a,int(artifact_meta["diffusion_order"]))
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=DCRNN(len(artifact_meta["sensor_ids"]),hidden_channels=artifact_meta["hidden_channels"],diffusion_order=artifact_meta["diffusion_order"],encoder_layers=artifact_meta["encoder_layers"],decoder_layers=artifact_meta["decoder_layers"],horizon=artifact_meta["forecast_horizon"])
        checkpoint=torch.load(ckpt_path,map_location=device,weights_only=False); model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval(); model_supports=(torch.tensor(f,device=device),torch.tensor(b,device=device)); trained_at=ckpt_path.stat().st_mtime
        persist_model_metadata()
        return True
    except Exception as exc:
        model=None; model_supports=None; artifact_meta=None; scaler=None; trained_at=None
        print(f"DCRNN artifact loading failed ({type(exc).__name__}); inference remains unavailable.")
        return False

@app.on_event("startup")
async def startup():
    global route_service,kafka_task
    if Path(cfg.dataset_path).is_file():
        try:
            loaded=dataset.load(cfg.dataset_path,cfg.zero_is_missing); route_service=RouteService(loaded.sensor_ids,cfg.sensor_metadata_path)
            sync_sensors()
        except Exception as exc: print(f"Dataset startup load failed: {exc}")
    load_model()
    if cfg.kafka_enabled:
        try:
            kafka_task=asyncio.create_task(consume_kafka())
            kafka_stats["enabled"]=True
        except Exception as exc:
            kafka_stats["last_error"]=type(exc).__name__

@app.on_event("shutdown")
async def shutdown():
    global kafka_consumer,kafka_task
    if kafka_task is not None:
        kafka_task.cancel()
        try: await kafka_task
        except asyncio.CancelledError: pass
    elif kafka_consumer is not None: await kafka_consumer.stop()

async def consume_kafka():
    global kafka_consumer
    try: from aiokafka import AIOKafkaConsumer
    except Exception as exc:
        kafka_stats.update(connected=False,last_error=type(exc).__name__)
        return
    while True:
        consumer=AIOKafkaConsumer(cfg.kafka_topic,bootstrap_servers=cfg.kafka_bootstrap_servers,group_id=cfg.kafka_consumer_group,enable_auto_commit=False,request_timeout_ms=5000)
        kafka_consumer=consumer
        try:
            await consumer.start(); kafka_stats.update(connected=True,last_error=None)
            async for message in consumer:
                try:
                    payload=json.loads(message.value.decode("utf-8")); item=ObservationItem.model_validate(payload)
                    if dataset.data is None or item.sensor_id not in dataset.data.sensor_ids: raise ValueError("Unknown sensor ID or dataset not loaded.")
                except Exception as exc:
                    kafka_stats["rejected"]+=1; kafka_stats["last_error"]=type(exc).__name__
                    await consumer.commit()
                    continue
                try:
                    from sqlalchemy.dialects.postgresql import insert
                    from app.db.session import SessionLocal
                    from app.db.models import Sensor,TrafficObservation
                    with SessionLocal.begin() as session:
                        if session.get(Sensor,item.sensor_id) is None: session.add(Sensor(sensor_id=item.sensor_id))
                        session.flush()
                        row={"sensor_id":item.sensor_id,"observed_at":item.observed_at,"speed":item.speed,"source":item.source or "kafka"}
                        statement=insert(TrafficObservation).values([row])
                        session.execute(statement.on_conflict_do_update(constraint="uq_observation_sensor_time",set_={"speed":statement.excluded.speed,"source":statement.excluded.source}))
                    kafka_stats["accepted"]+=1
                    record_ingestion([payload],"kafka")
                    dataset.append_observations([payload])
                except Exception as exc:
                    # Do not commit this offset: reconnecting will retry the durable write.
                    raise RuntimeError(f"Kafka observation persistence failed ({type(exc).__name__}).") from exc
                await consumer.commit()
        except asyncio.CancelledError: raise
        except Exception as exc:
            kafka_stats.update(connected=False,last_error=type(exc).__name__)
            print(f"Optional Kafka consumer reconnecting ({type(exc).__name__}); REST ingestion remains available.")
        finally:
            await consumer.stop(); kafka_consumer=None; kafka_stats["connected"]=False
        await asyncio.sleep(3)

@app.get("/api/v1/ingestion/status")
def ingestion_status(): return {"kafka":kafka_stats,"observations":observation_ingestion_stats,"rest_ingestion_available":True}

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/ready")
def ready():
    if dataset.data is None: raise HTTPException(503,"Dataset is not loaded")
    if model is None: raise HTTPException(503,"Dataset is loaded, but the trained DCRNN artifact is not ready. Run model training first.")
    return {"ready":True,"dataset_loaded":True,"model_ready":True}

@app.get("/api/v1/system/status")
def system_status():
    db="not_checked"
    try:
        from sqlalchemy import text
        from app.db.session import engine
        with engine.connect() as c: c.execute(text("SELECT 1")); db="connected"
    except Exception: db="unavailable"
    return {"database":db,"dataset":dataset.status(),"model_ready":model is not None,"architecture":"DCRNN","graph_available":bool(artifact_meta),"graph_source":artifact_meta.get("graph_source") if artifact_meta else None,"coordinates_available":bool(route_service and route_service.geo_available),"road_connectivity_available":bool(route_service and route_service.connectivity_available),"route_distance_available":bool(route_service and route_service.distance_available),"ingestion":{"kafka":kafka_stats,"observations":observation_ingestion_stats}}

@app.post("/api/v1/dataset/load")
def load_dataset():
    global route_service,observation_sources
    try:
        result=dataset.load(cfg.dataset_path,cfg.zero_is_missing); route_service=RouteService(result.sensor_ids,cfg.sensor_metadata_path)
        observation_sources={}
        observation_ingestion_stats.clear(); observation_ingestion_stats.update({"accepted":0,"transport_counts":{},"source_counts":{},"last_observed_at":None})
        sync_sensors()
        return dataset.status()
    except (OSError,ValueError) as exc: raise HTTPException(422,str(exc)) from exc

@app.post("/api/v1/dataset/upload")
async def upload_dataset(file: UploadFile=File(...),_:None=Depends(require_admin)):
    global route_service,observation_sources
    if not file.filename or Path(file.filename).suffix.lower() != ".csv": raise HTTPException(415,"Upload a CSV file.")
    cfg.artifact_dir.mkdir(parents=True,exist_ok=True); path=cfg.artifact_dir/"uploaded_dataset.csv"
    content=await file.read(100_000_001)
    if len(content)>100_000_000: raise HTTPException(413,"CSV exceeds 100 MB upload limit.")
    path.write_bytes(content)
    try:
        data=dataset.load(path,cfg.zero_is_missing); route_service=RouteService(data.sensor_ids,cfg.sensor_metadata_path)
        observation_sources={}
        observation_ingestion_stats.clear(); observation_ingestion_stats.update({"accepted":0,"transport_counts":{},"source_counts":{},"last_observed_at":None})
        sync_sensors()
        return dataset.status()
    except Exception as exc: raise HTTPException(422,str(exc)) from exc

@app.get("/api/v1/dataset/status")
def dataset_status(): return dataset.status()

@app.get("/api/v1/dataset/preview")
def dataset_preview(rows:int=Query(default=10,ge=1,le=100)):
    if dataset.data is None: raise HTTPException(409,"Load a dataset first.")
    records=[]
    for t,row in dataset.data.frame.iloc[:rows].iterrows():
        values={sid:(float(v) if np.isfinite(v) else None) for sid,v in row.items()}
        records.append({"timestamp":t.isoformat(),**values})
    return {"rows":records}

@app.get("/api/v1/dataset/statistics")
def dataset_statistics():
    if dataset.data is None: raise HTTPException(409,"Load a dataset first.")
    return {"rows":len(dataset.data.frame),"sensors":len(dataset.data.sensor_ids),"quality":dataset.data.quality,"interval_seconds":float(dataset.data.frame.index.to_series().diff().dt.total_seconds().median())}

@app.get("/api/v1/sensors")
def sensors(search:str|None=None,limit:int=Query(default=500,ge=1,le=1000),offset:int=Query(default=0,ge=0)):
    if dataset.data is None: raise HTTPException(409,"Load a dataset first.")
    ids=dataset.data.sensor_ids
    if search: ids=[s for s in ids if search.lower() in s.lower()]
    return {"items":[{"sensor_id":s,"node_index":dataset.data.sensor_ids.index(s)} for s in ids[offset:offset+min(limit,1000)]],"total":len(ids)}

@app.get("/api/v1/graph/locations")
def sensor_locations():
    if dataset.data is None: raise HTTPException(409,"Load a dataset first.")
    points=[]
    for sid in dataset.data.sensor_ids:
        coords=route_service.node_attributes.get(sid) if route_service else None
        if coords: points.append({"sensor_id":sid,**coords})
    return {"items":points,"count":len(points),"map_is_road_route":False}

@app.get("/api/v1/sensors/{sensor_id}")
def sensor(sensor_id:str):
    if dataset.data is None or sensor_id not in dataset.data.sensor_ids: raise HTTPException(404,"Unknown sensor ID.")
    coordinates=route_service.node_attributes.get(sensor_id) if route_service else None
    return {"sensor_id":sensor_id,"quality":dataset.data.quality[sensor_id],"coordinates":coordinates}

@app.get("/api/v1/sensors/{sensor_id}/history")
def sensor_history(sensor_id:str,limit:int=Query(default=1000,ge=1,le=10000)):
    if dataset.data is None or sensor_id not in dataset.data.sensor_ids: raise HTTPException(404,"Unknown sensor ID.")
    series=dataset.data.frame[sensor_id].tail(limit)
    offset=len(dataset.data.frame)-len(series); mask=dataset.data.observation_mask[offset:,dataset.data.sensor_ids.index(sensor_id)]
    return {"sensor_id":sensor_id,"unit":"mph","observations":[{"timestamp":t.isoformat(),"speed":float(v) if observed and np.isfinite(v) else None,"source":observation_sources.get(source_key(sensor_id,t.to_pydatetime()),"configured_csv") if observed else None} for (t,v),observed in zip(series.items(),mask)]}

@app.get("/api/v1/graph/correlations")
def sensor_correlations(sensor_ids:list[str]=Query(...,min_length=2,max_length=30)):
    if dataset.data is None: raise HTTPException(409,"Load a dataset first.")
    unknown=set(sensor_ids)-set(dataset.data.sensor_ids)
    if unknown: raise HTTPException(422,f"Unknown sensor IDs: {sorted(unknown)[:10]}")
    return {"sensor_ids":sensor_ids,"correlations":dataset.data.frame[sensor_ids].corr().fillna(0).values.tolist()}

@app.get("/api/v1/graph/status")
def graph_status():
    source=artifact_meta.get("graph_source") if artifact_meta else None
    is_road=source=="user-supplied road graph"
    message="DCRNN graph uses validated user-supplied road connectivity." if is_road else "Traffic-correlation graph supports are for forecasting only and do not represent roads."
    return {"available":bool(artifact_meta),"node_count":len(artifact_meta["sensor_ids"]) if artifact_meta else 0,"graph_source":source,"is_road_network":is_road,"road_connectivity_available":bool(route_service and route_service.connectivity_available),"coordinates_available":bool(route_service and route_service.geo_available),"road_distances_available":bool(route_service and route_service.distance_available),"message":message}

@app.get("/api/v1/graph/route")
def graph_route(source_id:str,destination_id:str,strategy:str="shortest_distance"):
    if not route_service: raise HTTPException(409,"Dataset is not loaded.")
    try:
        r=route_service.route(source_id,destination_id,strategy); return r.__dict__
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc

def predict_all(horizon:int):
    if dataset.data is None: raise HTTPException(409,"Dataset is not loaded.")
    if model is None or not artifact_meta: raise HTTPException(503,"DCRNN is not trained. Run python -m app.ml.train first.")
    if horizon>artifact_meta["forecast_horizon"]: raise HTTPException(422,f"Trained model supports at most {artifact_meta['forecast_horizon']} steps.")
    ids=artifact_meta["sensor_ids"]
    if ids != dataset.data.sensor_ids: raise HTTPException(409,"Loaded dataset sensor mapping differs from model artifacts; retrain or load the training dataset.")
    a=dataset.data.frame.to_numpy(np.float32)[-artifact_meta["input_window"]:]
    if len(a)<artifact_meta["input_window"]: raise HTTPException(422,"Insufficient observations for model input window.")
    x=(a-float(scaler["mean"]))/float(scaler["std"]); x=np.nan_to_num(x,nan=0.0); tensor=torch.tensor(x[None,:,:,None],dtype=torch.float32,device=next(model.parameters()).device)
    with torch.no_grad(): out=model(tensor,model_supports).cpu().numpy()[0,:horizon,: ,0]
    return float(scaler["mean"])+float(scaler["std"])*out

def departure_forecasts(req:TravelTimeRequest):
    if dataset.data is None: raise HTTPException(409,"Dataset is not loaded.")
    latest=dataset.data.frame.index[-1].to_pydatetime()
    departure=req.departure_time or latest
    if departure.tzinfo is not None: raise HTTPException(422,"Departure timestamp must use the dataset's timezone-naive timestamp convention.")
    delta=(departure-latest).total_seconds()
    if delta < 0: raise HTTPException(422,"Departure time cannot precede the latest loaded traffic observation.")
    interval=float(dataset.data.frame.index.to_series().diff().dt.total_seconds().median())
    first_step=max(0,int(np.ceil(delta/interval))-1)
    count=first_step+req.forecast_horizon
    values=predict_all(count)
    return values[first_step:count],departure,interval

@app.post("/api/v1/predictions")
def predictions(req:PredictionRequest):
    out=predict_all(req.horizon); ids=artifact_meta["sensor_ids"]
    selected=ids if req.sensor_ids is None else req.sensor_ids
    if not selected: raise HTTPException(422,"sensor_ids cannot be an empty list.")
    unknown=set(selected)-set(ids)
    if unknown: raise HTTPException(422,f"Unknown sensor IDs: {sorted(unknown)[:5]}")
    base=dataset.data.frame.index[-1]; interval=dataset.data.frame.index.to_series().diff().dt.total_seconds().median()
    result={"model_version":"dcrnn-best","units":"mph","predictions":[{"sensor_id":s,"forecast":[{"horizon_step":h+1,"forecast_time":(base+timedelta(seconds=float(interval)*(h+1))).isoformat(),"predicted_speed":float(out[h,ids.index(s)])} for h in range(req.horizon)]} for s in selected]}
    persist_predictions([{"sensor_id":p["sensor_id"],**point} for p in result["predictions"] for point in p["forecast"]])
    return result

@app.get("/api/v1/predictions/latest")
def latest_predictions(limit:int=Query(default=100,ge=1,le=1000)):
    try:
        from sqlalchemy import select
        from app.db.session import SessionLocal
        from app.db.models import Prediction
        with SessionLocal() as session:
            rows=session.scalars(select(Prediction).order_by(Prediction.generated_at.desc()).limit(limit)).all()
            return {"items":[{"prediction_id":r.id,"sensor_id":r.sensor_id,"generated_at":r.generated_at.isoformat(),"forecast_time":r.forecast_at.isoformat(),"predicted_speed":r.predicted_speed,"horizon_step":r.horizon_step,"model_version":r.model_version,"unit":"mph"} for r in rows]}
    except Exception as exc: raise HTTPException(503,f"Could not retrieve persisted predictions ({type(exc).__name__}).") from exc

@app.get("/api/v1/predictions/{prediction_id}")
def get_prediction(prediction_id:int):
    try:
        from app.db.session import SessionLocal
        from app.db.models import Prediction
        with SessionLocal() as session: row=session.get(Prediction,prediction_id)
        if row is None: raise HTTPException(404,"Prediction ID not found.")
        return {"prediction_id":row.id,"sensor_id":row.sensor_id,"generated_at":row.generated_at.isoformat(),"forecast_time":row.forecast_at.isoformat(),"predicted_speed":row.predicted_speed,"horizon_step":row.horizon_step,"model_version":row.model_version,"unit":"mph"}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(503,f"Could not retrieve prediction ({type(exc).__name__}).") from exc

@app.post("/api/v1/traffic/observations",status_code=201)
def ingest_observations(batch:ObservationBatch,_:None=Depends(require_admin)):
    if dataset.data is None: raise HTTPException(409,"Load a sensor registry before ingesting observations.")
    unknown={x.sensor_id for x in batch.observations}-set(dataset.data.sensor_ids)
    if unknown: raise HTTPException(422,f"Unknown sensor IDs: {sorted(unknown)[:10]}")
    try:
        from sqlalchemy.dialects.postgresql import insert
        from app.db.session import SessionLocal
        from app.db.models import Sensor,TrafficObservation
        with SessionLocal.begin() as session:
            for sid in {o.sensor_id for o in batch.observations}:
                session.merge(Sensor(sensor_id=sid))
            session.flush()
            unique={(o.sensor_id,o.observed_at):{"sensor_id":o.sensor_id,"observed_at":o.observed_at,"speed":o.speed,"source":o.source} for o in batch.observations}
            rows=list(unique.values())
            statement=insert(TrafficObservation).values(rows)
            session.execute(statement.on_conflict_do_update(constraint="uq_observation_sensor_time",set_={"speed":statement.excluded.speed,"source":statement.excluded.source}))
        dataset.append_observations(rows); record_ingestion(rows)
        return {"accepted":len(rows),"status":"persisted_and_inference_window_updated","sources":sorted({r["source"] for r in rows})}
    except Exception as exc: raise HTTPException(503,f"Could not persist observations ({type(exc).__name__}).") from exc

@app.post("/api/v1/travel-time/estimate")
def travel_time(req:TravelTimeRequest):
    if not route_service: raise HTTPException(409,"Load a dataset first.")
    try: route=route_service.route(req.source_sensor_id,req.destination_sensor_id,req.route_strategy)
    except ValueError as exc:
        message=str(exc)
        if "No validated road graph" in message:
            speeds,departure,interval=departure_forecasts(req)
            ids=artifact_meta["sensor_ids"]
            locations=route_service.node_attributes if route_service else {}
            points=[{"sensor_id":sensor_id,**locations[sensor_id]} for sensor_id in (req.source_sensor_id,req.destination_sensor_id) if sensor_id in locations]
            return {"source_sensor_id":req.source_sensor_id,"destination_sensor_id":req.destination_sensor_id,"route_sensor_ids":None,"distance_km":None,"estimated_duration":None,"predicted_sensor_speeds":{"source":[float(speeds[h,ids.index(req.source_sensor_id)]) for h in range(req.forecast_horizon)],"destination":[float(speeds[h,ids.index(req.destination_sensor_id)]) for h in range(req.forecast_horizon)],"unit":"mph"},"sensor_locations":points,"explanation":message,"departure_time":departure.isoformat(),"forecast_interval_seconds":interval,"forecast_horizon":req.forecast_horizon,"model_version":"dcrnn-best"}
        raise HTTPException(422,message) from exc
    speeds,departure,interval=departure_forecasts(req)
    sid=artifact_meta["sensor_ids"]
    speedmap={sensor:[float(speeds[h,i]) for h in range(req.forecast_horizon)] for i,sensor in enumerate(sid)}
    try: estimate=estimate_route_time(route,speedmap,"mph",interval)
    except ValueError as exc:
        if "distances for every route segment" in str(exc):
            return {"source_sensor_id":route.source_sensor_id,"destination_sensor_id":route.destination_sensor_id,"route_sensor_ids":route.sensor_ids,"distance_km":None,"estimated_duration":None,"predicted_sensor_speeds":{sensor:speedmap[sensor] for sensor in route.sensor_ids[1:]},"explanation":str(exc),"departure_time":departure.isoformat(),"forecast_interval_seconds":interval,"forecast_horizon":req.forecast_horizon,"model_version":"dcrnn-best"}
        raise HTTPException(422,str(exc)) from exc
    result={"source_sensor_id":route.source_sensor_id,"destination_sensor_id":route.destination_sensor_id,"route_sensor_ids":route.sensor_ids,"distance_km":route.total_distance_km,"estimated_duration":estimate,"departure_time":departure.isoformat(),"forecast_horizon":req.forecast_horizon,"model_version":"dcrnn-best","limitations":["Forecast speed is assigned from each segment's downstream sensor." ]}
    try:
        from app.db.session import SessionLocal
        from app.db.models import RouteRecord,TravelTimeEstimate
        with SessionLocal.begin() as session:
            session.add(RouteRecord(source_sensor=route.source_sensor_id,destination_sensor=route.destination_sensor_id,sensor_sequence=route.sensor_ids,graph_version=route.graph_source,distance_km=route.total_distance_km))
            session.add(TravelTimeEstimate(source_sensor=route.source_sensor_id,destination_sensor=route.destination_sensor_id,route=route.sensor_ids,distance_km=route.total_distance_km,estimated_seconds=estimate["estimated_seconds"],model_version="dcrnn-best"))
    except Exception as exc: print(f"Travel-time persistence unavailable ({type(exc).__name__}).")
    return result

@app.get("/api/v1/travel-time/latest")
def latest_travel_times(limit:int=Query(default=20,ge=1,le=100)):
    try:
        from sqlalchemy import select
        from app.db.session import SessionLocal
        from app.db.models import TravelTimeEstimate
        with SessionLocal() as session:
            rows=session.scalars(select(TravelTimeEstimate).order_by(TravelTimeEstimate.estimated_at.desc()).limit(limit)).all()
            return {"items":[{"id":r.id,"source_sensor_id":r.source_sensor,"destination_sensor_id":r.destination_sensor,"route_sensor_ids":r.route,"distance_km":r.distance_km,"estimated_seconds":r.estimated_seconds,"estimated_at":r.estimated_at.isoformat(),"model_version":r.model_version} for r in rows]}
    except Exception as exc: raise HTTPException(503,f"Could not load travel-time history ({type(exc).__name__}).") from exc

def run_training(job_id:str,overrides:dict):
    from app.ml.train import train
    try:
        jobs[job_id]={"job_id":job_id,"status":"running","progress":0}
        def update(record):
            history=jobs.get(job_id,{}).get("history",[]); history.append(record)
            jobs[job_id]={"job_id":job_id,"status":"running","progress":min(record["epoch"]/max(record["max_epochs"],1),.99),"latest_epoch":record,"history":history}
        metrics=train(progress_callback=update,overrides=overrides); load_model(); jobs[job_id]={"job_id":job_id,"status":"completed","progress":1,"metrics":metrics}; persist_job(job_id,"completed",1,metrics=metrics)
    except Exception as exc: jobs[job_id]={"job_id":job_id,"status":"failed","progress":0,"error":str(exc)}; persist_job(job_id,"failed",0,error=str(exc)[:1900])
    finally: training_lock.release()

@app.post("/api/v1/model/train",status_code=202)
def start_training(background:BackgroundTasks,request:TrainingRequest,_:None=Depends(require_admin)):
    if not training_lock.acquire(blocking=False): raise HTTPException(409,"A training job is already running.")
    job_id=str(uuid4()); jobs[job_id]={"job_id":job_id,"status":"queued","progress":0,"configuration":request.model_dump(exclude_none=True)}; persist_job(job_id,"queued",0); background.add_task(run_training,job_id,request.model_dump(exclude_none=True)); return jobs[job_id]

@app.get("/api/v1/model/train/{job_id}")
def training_status(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Unknown training job.")
    return jobs[job_id]

@app.get("/api/v1/model/metrics")
def model_metrics():
    path=cfg.artifact_dir/"metrics.json"
    if not path.is_file(): raise HTTPException(404,"Evaluation metrics are not available until training/evaluation has completed.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/v1/model/history")
def model_history():
    path=cfg.artifact_dir/"training_history.json"
    if not path.is_file(): raise HTTPException(404,"Training history is unavailable until training completes.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/v1/model/test-sample")
def model_test_sample(limit:int=Query(default=5000,ge=1,le=20000)):
    path=cfg.artifact_dir/"test_predictions_sample.csv"
    if not path.is_file(): raise HTTPException(404,"Held-out sample predictions are unavailable until training completes.")
    import pandas as pd
    return {"rows":pd.read_csv(path,nrows=limit).to_dict("records")}

@app.get("/api/v1/model/metadata")
def model_metadata():
    if not artifact_meta: raise HTTPException(503,"No trained DCRNN artifacts are available.")
    return {k:v for k,v in artifact_meta.items() if k not in {"adjacency"}} | {"architecture":"DCRNN"}

@app.post("/api/v1/model/evaluate")
def evaluate_model():
    path=cfg.artifact_dir/"metrics.json"
    if not path.exists(): raise HTTPException(503,"No test evaluation is available. Run python -m app.ml.train.")
    return json.loads(path.read_text(encoding="utf-8"))

