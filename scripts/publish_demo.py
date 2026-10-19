"""Publish explicitly simulated demo observations to the configured Kafka topic."""
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import json
from aiokafka import AIOKafkaProducer
from app.core.config import get_settings
from app.ml.preprocessing import load_traffic_csv

async def main():
    parser=argparse.ArgumentParser(description="Send clearly labelled synthetic demo observations; this is not live traffic.")
    parser.add_argument("--sensor-id",required=True,help="An actual sensor ID present in DATASET_PATH")
    parser.add_argument("--speed-mph",type=float,default=35.0)
    parser.add_argument("--count",type=int,default=1)
    parser.add_argument("--bootstrap-servers",default=None,help="Defaults to KAFKA_BOOTSTRAP_SERVERS")
    args=parser.parse_args(); cfg=get_settings()
    data=load_traffic_csv(cfg.dataset_path,cfg.zero_is_missing,cfg.dataset_timestamp_column,cfg.dataset_sensor_id_column,cfg.dataset_value_column)
    if args.sensor_id not in data.sensor_ids: raise SystemExit("Sensor ID is not present in configured dataset.")
    if not 0<=args.speed_mph<=200 or not 1<=args.count<=1000: raise SystemExit("Use speed 0..200 mph and count 1..1000.")
    producer=AIOKafkaProducer(bootstrap_servers=args.bootstrap_servers or cfg.kafka_bootstrap_servers)
    await producer.start()
    try:
        for i in range(args.count):
            payload={"sensor_id":args.sensor_id,"observed_at":(datetime.now(timezone.utc)+timedelta(minutes=5*i)).replace(tzinfo=None).isoformat(),"speed":args.speed_mph,"source":"synthetic-demo"}
            await producer.send_and_wait(cfg.kafka_topic,json.dumps(payload).encode("utf-8"))
        print(f"Published {args.count} SYNTHETIC demo observation(s); these are not live traffic data.")
    finally: await producer.stop()

if __name__=="__main__": asyncio.run(main())
