import pytest
pytest.importorskip("pydantic_settings")
import pandas as pd
from app.ml.preprocessing import TrafficData
from app.services.dataset_service import DatasetService

def test_stream_observations_extend_inference_window_and_mask():
    index=pd.date_range("2024-01-01",periods=2,freq="5min")
    frame=pd.DataFrame({"001":[10.0,11.0],"002":[20.0,21.0]},index=index,dtype="float32")
    quality={sid:{"missing_proportion":0.0,"zero_proportion":0.0,"invalid_proportion":0.0} for sid in frame.columns}
    data=TrafficData(frame,index,["001","002"],quality,"timestamp",frame.notna().to_numpy())
    service=DatasetService(); service.data=data
    service.append_observations([{"sensor_id":"001","observed_at":index[-1]+pd.Timedelta(minutes=5),"speed":12.0,"source":"test"}])
    assert len(service.data.frame)==3
    assert service.data.frame.iloc[-1]["001"]==12
    assert service.data.frame.iloc[-1]["002"]==21
    assert service.data.observation_mask[-1].tolist()==[True,False]
