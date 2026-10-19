import pandas as pd
import numpy as np
import pytest
from app.ml.preprocessing import load_traffic_csv, TrainOnlyScaler, make_windows

def test_wide_csv_detection_sorts_and_aggregates_duplicates(tmp_path):
    path=tmp_path/"traffic.csv"
    pd.DataFrame({"Unnamed: 0":["2024-01-01 00:05","2024-01-01 00:00","2024-01-01 00:00"],"773869":[2,1,3],"name":["x","y","z"]}).to_csv(path,index=False)
    data=load_traffic_csv(path,zero_is_missing=False)
    assert data.sensor_ids==["773869"]
    assert list(data.frame.index)==sorted(data.frame.index)
    assert data.frame.iloc[0,0]==2

def test_bad_schema_has_actionable_error(tmp_path):
    path=tmp_path/"bad.csv"; pd.DataFrame({"a":["not a date"],"b":[1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match="timestamp"):
        load_traffic_csv(path)

def test_long_csv_mapping_preserves_sensor_ids_and_averages_duplicates(tmp_path):
    path=tmp_path/"long.csv"
    pd.DataFrame({"when":["2024-01-01 00:00","2024-01-01 00:00","2024-01-01 00:05"],"detector":["001","001","001"],"velocity":[10,20,30]}).to_csv(path,index=False)
    data=load_traffic_csv(path,sensor_id_column="detector",value_column="velocity")
    assert data.sensor_ids==["001"]
    assert list(data.frame.index)==sorted(data.frame.index)
    assert data.frame.iloc[0,0]==15

def test_invalid_duplicate_readings_are_removed_before_aggregation(tmp_path):
    path=tmp_path/"invalid_duplicate.csv"
    pd.DataFrame({"timestamp":["2024-01-01","2024-01-01"],"sensor-A":[300,50]}).to_csv(path,index=False)
    data=load_traffic_csv(path)
    assert data.frame.iloc[0,0]==50
    assert data.quality["sensor-A"]["invalid_proportion"]==pytest.approx(.5)

def test_scaler_is_fit_from_training_values_only():
    scaler=TrainOnlyScaler().fit(np.array([[1.,3.],[2.,4.]]))
    assert scaler.mean_==2.5
    assert np.allclose(scaler.inverse_transform(scaler.transform(np.array([9.]))),[9.])

def test_windows_have_expected_shape_and_target_mask():
    x=np.arange(20*2,dtype=np.float32).reshape(20,2)
    xs,ys,mask=make_windows(x,4,3)
    assert xs.shape==(14,4,2,1); assert ys.shape==(14,3,2,1); assert mask.all()

