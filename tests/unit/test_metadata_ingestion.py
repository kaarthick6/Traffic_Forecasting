import pandas as pd
import pytest
from app.services.route_service import RouteService

def test_local_sensor_location_metadata_has_coordinates_but_no_road_route(tmp_path):
    path=tmp_path/"locations.csv"
    pd.DataFrame({"sensor_id":["001","002"],"latitude":[34.0,34.1],"longitude":[-118.0,-118.1]}).to_csv(path,index=False)
    service=RouteService(["001","002"],path)
    assert service.geo_available is True
    assert len(service.node_attributes)==2
    assert service.connectivity_available is False
    assert service.distance_available is False
    with pytest.raises(ValueError,match="road graph"):
        service.route("001","002")

def test_coordinate_metadata_rejects_sensor_ids_not_in_dataset(tmp_path):
    path=tmp_path/"locations.csv"
    pd.DataFrame({"sensor_id":["outside"],"latitude":[34.0],"longitude":[-118.0]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match="absent from dataset"):
        RouteService(["001"],path)
