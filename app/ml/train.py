"""Reproducible DCRNN training entry point: python -m app.ml.train"""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.core.config import get_settings
from app.ml.preprocessing import load_traffic_csv, TrainOnlyScaler, make_windows
from app.ml.graph import correlation_adjacency, road_adjacency, diffusion_supports
from app.ml.dcrnn import DCRNN, masked_mae


def _train(progress_callback=None, overrides:dict|None=None) -> dict:
    cfg=get_settings().model_copy(update={k:v for k,v in (overrides or {}).items() if v is not None})
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    data=load_traffic_csv(cfg.dataset_path,cfg.zero_is_missing,cfg.dataset_timestamp_column,cfg.dataset_sensor_id_column,cfg.dataset_value_column); values=data.frame.to_numpy(np.float32)
    n=len(values); train_end=int(n*.7); val_end=int(n*.8)
    if train_end < cfg.input_window+cfg.forecast_horizon: raise ValueError("Dataset has too few rows for configured train windows.")
    train_raw=values[:train_end]; observed=data.observation_mask[:train_end]
    scaler=TrainOnlyScaler().fit(train_raw,observed)
    # Interpolated frame values are used for model inputs, while genuine observation mask is retained for targets.
    scaled=np.nan_to_num(scaler.transform(values),nan=0.0)[...,None]
    valid=data.observation_mask[...,None]
    graph_values=np.where(np.isfinite(train_raw),scaler.mean_,train_raw)
    graph_source="training-data pairwise speed correlation; forecasting only, not a road network"
    if cfg.sensor_metadata_path:
        from app.services.route_service import RouteService
        supplied=RouteService(data.sensor_ids,cfg.sensor_metadata_path)
        if supplied.connectivity_available:
            edges=[{"source_sensor":source,"destination_sensor":target,**attrs} for source,neighbors in supplied.graph.items() for target,attrs in neighbors.items()]
            a=road_adjacency(data.sensor_ids,edges); graph_source="user-supplied road graph"
        else: a=correlation_adjacency(graph_values,cfg.graph_top_k,cfg.graph_correlation_threshold)
    else: a=correlation_adjacency(graph_values,cfg.graph_top_k,cfg.graph_correlation_threshold)
    f,b=diffusion_supports(a,cfg.diffusion_order)
    xs=[]; ys=[]; ms=[]
    # Split samples by target end time, ensuring no validation/test values leak into fitting.
    for start in range(0, n-cfg.input_window-cfg.forecast_horizon+1):
        target_end=start+cfg.input_window+cfg.forecast_horizon
        if target_end <= train_end: part="train"
        elif start+cfg.input_window >= train_end and target_end <= val_end: part="val"
        elif start+cfg.input_window >= val_end: part="test"
        else: continue
        xs.append((part,scaled[start:start+cfg.input_window])); ys.append((part,scaled[start+cfg.input_window:target_end])); ms.append((part,valid[start+cfg.input_window:target_end]))
    def pack(items,part): return (np.stack([x for p,x in items if p==part]),np.stack([x for p,x in ys if p==part]),np.stack([x for p,x in ms if p==part]))
    tr=pack(xs,"train"); va=pack(xs,"val"); te=pack(xs,"test")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supports=(torch.tensor(f,device=device),torch.tensor(b,device=device))
    model=DCRNN(len(data.sensor_ids),hidden_channels=cfg.hidden_channels,diffusion_order=cfg.diffusion_order,encoder_layers=cfg.encoder_layers,decoder_layers=cfg.decoder_layers,horizon=cfg.forecast_horizon).to(device)
    def loader(part,shuffle):
        x,y,m=[torch.from_numpy(z) for z in part]
        return DataLoader(TensorDataset(x,y,m),batch_size=cfg.batch_size,shuffle=shuffle)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate); best=float("inf"); wait=0; history=[]; started=time.time()
    cfg.artifact_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,cfg.max_epochs+1):
        model.train(); losses=[]
        for x,y,m in loader(tr,True):
            x,y,m=x.to(device),y.to(device),m.to(device); optimizer.zero_grad(); pred=model(x,supports,y,teacher_forcing=max(0.5*(1-epoch/cfg.max_epochs),0)); loss=masked_mae(pred,y,m); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); vloss=[]
        with torch.no_grad():
            for x,y,m in loader(va,False): vloss.append(float(masked_mae(model(x.to(device),supports),y.to(device),m.to(device))))
        val_loss=float(np.mean(vloss)); record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.time()-started}; history.append(record); print(json.dumps(record),flush=True)
        if progress_callback: progress_callback({"epoch":epoch,"max_epochs":cfg.max_epochs,"train_loss":record["train_loss"],"validation_loss":val_loss,"learning_rate":record["learning_rate"],"elapsed_seconds":record["elapsed_seconds"]})
        if val_loss < best:
            best=val_loss; wait=0; torch.save({"state_dict":model.state_dict(),"model_config":{"num_nodes":len(data.sensor_ids),"input_dim":1,"output_dim":1,"hidden_channels":cfg.hidden_channels,"diffusion_order":cfg.diffusion_order,"encoder_layers":cfg.encoder_layers,"decoder_layers":cfg.decoder_layers,"horizon":cfg.forecast_horizon}},cfg.artifact_dir/"dcrnn.pt")
        else:
            wait+=1
            if wait >= cfg.early_stopping_patience: break
    median_interval=float(data.timestamps.to_series().diff().dt.total_seconds().median())
    artifacts={"sensor_ids":data.sensor_ids,"sensor_to_index":{sensor:i for i,sensor in enumerate(data.sensor_ids)},"sensor_quality":data.quality,"feature_config":{"input_features":["traffic_speed"],"output_features":["traffic_speed"],"units":"mph"},"sampling_interval_seconds":median_interval,"scaler":scaler.to_dict(),"adjacency":a.tolist(),"graph_source":graph_source,"graph_top_k":cfg.graph_top_k,"graph_correlation_threshold":cfg.graph_correlation_threshold,"input_window":cfg.input_window,"forecast_horizon":cfg.forecast_horizon,"diffusion_order":cfg.diffusion_order,"hidden_channels":cfg.hidden_channels,"encoder_layers":cfg.encoder_layers,"decoder_layers":cfg.decoder_layers,"batch_size":cfg.batch_size,"learning_rate":cfg.learning_rate,"max_epochs":cfg.max_epochs,"early_stopping_patience":cfg.early_stopping_patience,"seed":42,"dataset_path":str(Path(cfg.dataset_path).name),"dataset_rows":len(data.frame),"timestamp_start":data.timestamps[0].isoformat(),"timestamp_end":data.timestamps[-1].isoformat()}
    (cfg.artifact_dir/"metadata.json").write_text(json.dumps(artifacts,indent=2),encoding="utf-8")
    (cfg.artifact_dir/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    model.load_state_dict(torch.load(cfg.artifact_dir/"dcrnn.pt",map_location=device,weights_only=False)["state_dict"]); model.eval()
    pred_all=[]; actual_all=[]; masks_all=[]
    with torch.no_grad():
        for x,y,m in loader(te,False): pred_all.append(model(x.to(device),supports).cpu().numpy()); actual_all.append(y.numpy()); masks_all.append(m.numpy())
    pred=scaler.inverse_transform(np.concatenate(pred_all)); actual=scaler.inverse_transform(np.concatenate(actual_all)); mask=np.concatenate(masks_all).astype(bool)
    by_horizon=[]
    for h in range(pred.shape[1]):
        ok=mask[:,h,...,0]; err=pred[:,h,...,0][ok]-actual[:,h,...,0][ok]
        by_horizon.append({"horizon_step":h+1,"mae":float(np.mean(np.abs(err))),"mse":float(np.mean(err**2)),"rmse":float(np.sqrt(np.mean(err**2)))})
    all_err=(pred[...,0]-actual[...,0])[mask[...,0]]
    per_sensor=[]
    for i,sensor_id in enumerate(data.sensor_ids):
        ok=mask[:,:,i,0]; errors=(pred[:,:,i,0]-actual[:,:,i,0])[ok]
        per_sensor.append({"sensor_id":sensor_id,"mae":float(np.mean(np.abs(errors))),"mse":float(np.mean(errors**2)),"rmse":float(np.sqrt(np.mean(errors**2))),"valid_targets":int(ok.sum())})
    metrics={"units":"mph","mae":float(np.mean(np.abs(all_err))),"mse":float(np.mean(all_err**2)),"rmse":float(np.sqrt(np.mean(all_err**2))),"horizon_metrics":by_horizon,"sensor_metrics":per_sensor,"test_targets":int(mask.sum())}
    (cfg.artifact_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    sample=[]
    for i,sensor_id in enumerate(data.sensor_ids):
        valid_idx=np.flatnonzero(mask[:,0,i,0])[:1000]
        sample.extend({"sensor_id":sensor_id,"horizon_step":1,"actual_speed_mph":float(actual[j,0,i,0]),"predicted_speed_mph":float(pred[j,0,i,0])} for j in valid_idx)
    pd.DataFrame(sample).to_csv(cfg.artifact_dir/"test_predictions_sample.csv",index=False)
    return metrics


def train(progress_callback=None, overrides:dict|None=None) -> dict:
    from filelock import FileLock, Timeout
    artifact_dir=get_settings().artifact_dir
    artifact_dir.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(artifact_dir/"training.lock"),timeout=0):
            return _train(progress_callback,overrides)
    except Timeout as exc:
        raise RuntimeError("Another training process is already writing this artifact directory.") from exc


if __name__ == "__main__": print(json.dumps(train(),indent=2))

