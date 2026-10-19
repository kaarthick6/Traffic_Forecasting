"""Road route and physical travel-time estimation with strict metadata checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import heapq
import pandas as pd
try:
    import networkx as nx
except ImportError:  # Standard-library fallback keeps metadata validation usable offline.
    nx = None


@dataclass
class RouteResult:
    source_sensor_id: str
    destination_sensor_id: str
    sensor_ids: list[str]
    edges: list[dict]
    total_distance_km: float | None
    strategy: str
    graph_source: str


class RouteService:
    def __init__(self, sensor_ids: list[str], metadata_path: str | Path | None = None):
        self.sensor_ids = set(map(str, sensor_ids)); self.graph: dict[str,dict[str,dict]] = {sid:{} for sid in self.sensor_ids}; self.node_attributes={sid:{} for sid in self.sensor_ids}
        self.graph_source = "no road-network metadata"; self.distance_available = False; self.geo_available = False; self.connectivity_available = False
        if metadata_path:
            self.load_metadata(metadata_path)

    def load_metadata(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(f"Sensor/road metadata not found: {path}")
        if path.suffix.lower() == ".json":
            doc=json.loads(path.read_text(encoding="utf-8")); edges=pd.DataFrame(doc.get("edges", [])); sensors=pd.DataFrame(doc.get("sensors", []))
        else:
            table=pd.read_csv(path,dtype={"sensor_id":str})
            if {"sensor_id","latitude","longitude"}.issubset(table.columns):
                sensors=table; edges=pd.DataFrame()
            else:
                edges=table; sensors=pd.DataFrame()
        coordinates = {}
        if not sensors.empty and {"sensor_id","latitude","longitude"}.issubset(sensors.columns):
            if sensors["sensor_id"].duplicated().any(): raise ValueError("Sensor coordinate metadata contains duplicate sensor_id entries.")
            for row in sensors.to_dict("records"):
                sid=str(row["sensor_id"])
                if sid not in self.sensor_ids: raise ValueError(f"Coordinate metadata references sensor absent from dataset: {sid}")
                if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                    lat,lon=float(row["latitude"]),float(row["longitude"])
                    if not (-90<=lat<=90 and -180<=lon<=180): raise ValueError(f"Invalid WGS84 coordinates for sensor {sid}.")
                    coordinates[sid]={"latitude":lat,"longitude":lon}
            self.geo_available=bool(coordinates)
        self.node_attributes=coordinates
        required={"source_sensor","destination_sensor"}
        if not required.issubset(edges.columns):
            # Also support source_sensor_id/destination_sensor_id names.
            edges=edges.rename(columns={"source_sensor_id":"source_sensor","destination_sensor_id":"destination_sensor"})
        if not required.issubset(edges.columns):
            # Sensor coordinates by themselves can support a map, but not a road route.
            self.graph_source="sensor coordinates only; road connectivity unavailable"
            self.distance_available=False
            return
        self.graph={sid:{} for sid in self.sensor_ids}; self.node_attributes=coordinates
        for row in edges.to_dict("records"):
            src,dst=str(row["source_sensor"]),str(row["destination_sensor"])
            if src not in self.sensor_ids or dst not in self.sensor_ids: raise ValueError(f"Road metadata references sensor absent from dataset: {src} or {dst}")
            attrs={}
            if "edge_distance_km" in row and pd.notna(row.get("edge_distance_km")):
                dist=float(row["edge_distance_km"])
                if not math.isfinite(dist) or dist <= 0: raise ValueError("Road edge distances must be finite positive kilometres.")
                attrs["distance_km"]=dist
            if pd.notna(row.get("edge_travel_time_seconds")):
                sec=float(row["edge_travel_time_seconds"])
                if sec > 0: attrs["travel_time_seconds"]=sec
            self.graph[src][dst]=attrs
        self.connectivity_available=any(self.graph.values())
        self.distance_available=self.connectivity_available and all("distance_km" in d for neighbors in self.graph.values() for d in neighbors.values())
        self.graph_source="user-supplied road-edge metadata" if self.connectivity_available else "no validated road distances"

    def route(self, source: str, destination: str, strategy: str="shortest_distance") -> RouteResult:
        source,destination=str(source),str(destination)
        if source not in self.sensor_ids or destination not in self.sensor_ids: raise ValueError("Source and destination must be sensor IDs in the loaded dataset.")
        if source == destination: raise ValueError("Source and destination must be different sensors.")
        if not getattr(self,"connectivity_available",False): raise ValueError("No validated road graph is configured. Supply road connectivity; correlation edges are not driving routes.")
        if strategy not in {"shortest_distance","fastest"}: raise ValueError("route_strategy must be shortest_distance or fastest.")
        weight="distance_km" if strategy=="shortest_distance" else "travel_time_seconds"
        if strategy=="fastest" and any(weight not in d for neighbors in self.graph.values() for d in neighbors.values()):
            raise ValueError("Fastest-route strategy requires edge_travel_time_seconds on every candidate road edge.")
        if strategy=="shortest_distance" and not self.distance_available: weight=None
        if nx is not None:
            graph=nx.DiGraph(); graph.add_nodes_from(self.sensor_ids)
            for node,neighbors in self.graph.items():
                for neighbor,attrs in neighbors.items(): graph.add_edge(node,neighbor,**attrs)
            try: ids=nx.shortest_path(graph,source,destination,weight=weight)
            except nx.NetworkXNoPath as exc: raise ValueError(f"No validated road path from {source} to {destination}.") from exc
        else:
            costs={source:0.0}; previous={}; queue=[(0.0,source)]
            while queue:
                cost,node=heapq.heappop(queue)
                if cost!=costs.get(node): continue
                for neighbor,attrs in self.graph.get(node,{}).items():
                    edge_cost=float(attrs[weight]) if weight else 1.0; new_cost=cost+edge_cost
                    if new_cost<costs.get(neighbor,math.inf): costs[neighbor]=new_cost; previous[neighbor]=node; heapq.heappush(queue,(new_cost,neighbor))
            if destination not in costs: raise ValueError(f"No validated road path from {source} to {destination}.")
            ids=[destination]
            while ids[-1]!=source: ids.append(previous[ids[-1]])
            ids.reverse()
        effective_strategy="topological" if strategy=="shortest_distance" and not self.distance_available else strategy
        edges=[]; total=0.0
        for a,b in zip(ids,ids[1:]):
            d=self.graph[a][b]; distance=d.get("distance_km"); edges.append({"source_sensor_id":a,"destination_sensor_id":b,"distance_km":distance})
            if distance is not None: total+=distance
        return RouteResult(source,destination,ids,edges,total if self.distance_available else None,effective_strategy,self.graph_source)


def estimate_route_time(route: RouteResult, predicted_speeds: dict[str,float | list[float]], unit: str="mph", forecast_interval_seconds: float=300) -> dict:
    if route.total_distance_km is None or not route.edges: raise ValueError("Physical travel time requires validated positive distances for every route segment.")
    if unit.lower() not in {"mph","mi/h","km/h","kph"}: raise ValueError("Speed unit must be mph or km/h.")
    factor=1.609344 if unit.lower() in {"mph","mi/h"} else 1.0
    if forecast_interval_seconds <= 0: raise ValueError("forecast_interval_seconds must be positive.")
    segments=[]; elapsed_total=0.0; extrapolated_any=False
    for edge in route.edges:
        # Documented segment mapping: use the downstream sensor's forecast speed.
        sid=edge["destination_sensor_id"]; raw=predicted_speeds.get(sid)
        speeds=list(raw) if isinstance(raw,(list,tuple)) else ([float(raw)] if raw is not None else [])
        if not speeds or any(not math.isfinite(float(v)) or float(v)<=0 or float(v)>200 for v in speeds):
            raise ValueError(f"Cannot estimate segment {edge['source_sensor_id']}→{sid}: predicted speed is missing or implausible.")
        remaining=float(edge["distance_km"]); segment_seconds=0.0; used=[]; extrapolated=False
        while remaining > 1e-12:
            horizon_index=int(elapsed_total//forecast_interval_seconds)
            if horizon_index >= len(speeds): horizon_index=len(speeds)-1; extrapolated=True
            speed=float(speeds[horizon_index]); km_per_second=speed*factor/3600
            interval_left=forecast_interval_seconds-(elapsed_total%forecast_interval_seconds)
            travel_seconds=remaining/km_per_second
            portion=min(travel_seconds,interval_left)
            traveled=portion*km_per_second
            remaining-=traveled; elapsed_total+=portion; segment_seconds+=portion; used.append(speed)
        extrapolated_any |= extrapolated
        segments.append({**edge,"sensor_id_used_for_speed":sid,"predicted_speeds":speeds,"speed_used_by_forecast_interval":used,"speed_unit":unit,"estimated_seconds":segment_seconds,"estimated_minutes":segment_seconds/60,"forecast_extrapolated":extrapolated})
    return {"estimated_seconds":elapsed_total,"estimated_minutes":elapsed_total/60,"estimated_hours":elapsed_total/3600,"segments":segments,"speed_unit":unit,"forecast_interval_seconds":forecast_interval_seconds,"segment_speed_mapping":"downstream sensor speed","extrapolated_beyond_forecast":extrapolated_any,"extrapolation_method":"reuse the final available horizon speed for later intervals"}

