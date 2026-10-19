from app.ml.preprocessing import load_traffic_csv
from app.core.config import get_settings
import pandas as pd

class DatasetService:
    def __init__(self): self.data=None
    def load(self,path,zero_is_missing=True):
        cfg=get_settings()
        self.data=load_traffic_csv(path,zero_is_missing,cfg.dataset_timestamp_column,cfg.dataset_sensor_id_column,cfg.dataset_value_column)
        return self.data
    def status(self):
        if self.data is None: return {"loaded":False,"validation_status":"not_loaded"}
        return {"loaded":True,"validation_status":"valid","rows":len(self.data.frame),"sensors":len(self.data.sensor_ids),"start":self.data.timestamps[0].isoformat(),"end":self.data.timestamps[-1].isoformat(),"sensor_ids":self.data.sensor_ids,"quality":self.data.quality}

    def append_observations(self,observations):
        """Update the in-memory inference window after durable REST/Kafka ingestion."""
        if self.data is None: raise ValueError("Load a sensor registry before appending observations.")
        rows=[{"timestamp":pd.Timestamp(o["observed_at"]),"sensor_id":str(o["sensor_id"]),"speed":float(o["speed"])} for o in observations]
        updates=pd.DataFrame(rows).pivot_table(index="timestamp",columns="sensor_id",values="speed",aggfunc="mean")
        updates=updates.reindex(columns=self.data.sensor_ids)
        frame=pd.concat([self.data.frame,updates]).sort_index(kind="stable").groupby(level=0,sort=True).last()
        existing_mask=pd.DataFrame(self.data.observation_mask,index=self.data.frame.index,columns=self.data.sensor_ids)
        update_mask=updates.notna()
        mask=pd.concat([existing_mask,update_mask]).sort_index(kind="stable").groupby(level=0,sort=True).max()
        self.data.frame=frame.ffill().astype("float32")
        self.data.timestamps=self.data.frame.index
        self.data.observation_mask=mask.to_numpy(dtype=bool)
        for sensor_id in self.data.sensor_ids:
            self.data.quality[sensor_id]["missing_proportion"]=float((~mask[sensor_id]).mean())
            self.data.quality[sensor_id]["zero_proportion"]=float((frame[sensor_id]==0).mean())

