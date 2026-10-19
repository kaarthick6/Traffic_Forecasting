import numpy as np
import pandas as pd
import pytest
from app.ml.graph import correlation_adjacency, diffusion_supports, road_adjacency
from app.services.route_service import RouteService, RouteResult, estimate_route_time

def test_correlation_adjacency_and_diffusion_are_safe():
    values=np.array([[1,1,5],[2,2,5],[3,3,5],[4,4,5]],dtype=float)
    a=correlation_adjacency(values,top_k=1,threshold=.8)
    assert a.shape==(3,3); assert np.allclose(a,a.T)
    f,b=diffusion_supports(a,2)
    assert f.shape==(3,3,3); assert b.shape==f.shape
    isolated=np.zeros((2,2)); sf,sb=diffusion_supports(isolated,2)
    assert np.isfinite(sf).all() and np.isfinite(sb).all()

def test_road_adjacency_uses_supplied_direction_and_mapping():
    a=road_adjacency(["sensor-X","sensor-Y"],[{"source_sensor":"sensor-X","destination_sensor":"sensor-Y","edge_distance_km":2.0}])
    assert a[0,1]==pytest.approx(.5) and a[1,0]==0

def test_missing_road_distance_blocks_route_time(tmp_path):
    path=tmp_path/"edges.csv"
    pd.DataFrame({"source_sensor":["a"],"destination_sensor":["b"]}).to_csv(path,index=False)
    service=RouteService(["a","b"],path)
    route=service.route("a","b")
    assert route.sensor_ids==["a","b"] and route.total_distance_km is None
    with pytest.raises(ValueError,match="positive distances"):
        estimate_route_time(route,{"b":30})

def test_no_road_graph_and_missing_speed_are_rejected():
    service=RouteService(["a","b"])
    with pytest.raises(ValueError,match="road graph"):
        service.route("a","b")
    route=RouteResult("a","b",["a","b"],[{"source_sensor_id":"a","destination_sensor_id":"b","distance_km":1.0}],1.0,"shortest_distance","test")
    with pytest.raises(ValueError,match="missing or implausible"):
        estimate_route_time(route,{"b":0})

def test_road_route_and_mph_conversion(tmp_path):
    path=tmp_path/"edges.csv"
    pd.DataFrame({"source_sensor":["a","b"],"destination_sensor":["b","c"],"edge_distance_km":[1.609344,1.609344]}).to_csv(path,index=False)
    service=RouteService(["a","b","c"],path); route=service.route("a","c")
    assert route.sensor_ids==["a","b","c"] and route.total_distance_km==pytest.approx(3.218688)
    result=estimate_route_time(route,{"b":60,"c":60},"mph")
    assert result["estimated_minutes"]==pytest.approx(2)

def test_invalid_sensor_and_same_endpoint_rejected():
    service=RouteService(["a","b"])
    with pytest.raises(ValueError,match="dataset"): service.route("x","b")
    with pytest.raises(ValueError,match="different"): service.route("a","a")

def test_forecast_speed_changes_by_interval_and_extrapolation_is_reported():
    route=RouteResult("a","b",["a","b"],[{"source_sensor_id":"a","destination_sensor_id":"b","distance_km":20.0}],20.0,"shortest_distance","test")
    result=estimate_route_time(route,{"b":[60,30]},"km/h",300)
    assert result["extrapolated_beyond_forecast"] is True
    assert result["segments"][0]["speed_used_by_forecast_interval"]==[60.0,30.0,30.0]

