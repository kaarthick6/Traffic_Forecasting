from __future__ import annotations
import os
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_app.services.api_client import get, post, API_BASE_URL

st.set_page_config(page_title="DCRNN Traffic Travel-Time",page_icon="🚦",layout="wide")
st.title("🚦 Traffic flow & travel-time estimator")
st.caption("DCRNN forecasts sensor speeds. Physical travel duration requires a validated road graph and edge distances.")

pages=["Travel-Time Estimator","Dashboard Overview","Dataset Explorer","Sensor Analysis","Model Training","Model Evaluation","Graph & Route Diagnostics","System Health & Settings"]
page=st.sidebar.radio("Workspace",pages)

def api_error(exc):
    if isinstance(exc,httpx.HTTPStatusError):
        try: st.error(exc.response.json().get("detail",str(exc)))
        except Exception: st.error(str(exc))
    else: st.error(f"API unavailable at {API_BASE_URL}: {exc}")

if page=="Travel-Time Estimator":
    st.header("Estimate a trip between sensors")
    try:
        registry=get("/api/v1/sensors",limit=1000)
        ids=[x["sensor_id"] for x in registry["items"]]
        if len(ids)<2: st.info("Load a dataset with at least two sensors to explore forecasts.")
        else:
            c1,c2=st.columns(2)
            src=c1.selectbox("Source sensor",ids,key="source")
            destinations=[x for x in ids if x!=src]
            dst=c2.selectbox("Destination sensor",destinations,key="destination")
            c3,c4=st.columns(2)
            departure=c3.datetime_input("Optional departure time",value=None)
            horizon=c4.selectbox("Forecast horizon",[1,3,6,12],index=0)
            strategy=st.selectbox("Route strategy",["shortest_distance","fastest"])
            if st.button("Estimate travel time",type="primary"):
                body={"source_sensor_id":src,"destination_sensor_id":dst,"forecast_horizon":horizon,"route_strategy":strategy}
                if departure: body["departure_time"]=departure.isoformat()
                result=post("/api/v1/travel-time/estimate",body)
                st.subheader("Trip estimate")
                if result.get("estimated_duration") is None:
                    st.warning(result.get("explanation","Physical travel time is unavailable without validated road metadata."))
                    st.info("No route or distance has been fabricated. DCRNN sensor forecasts remain available.")
                    if result.get("route_sensor_ids"):
                        st.write("Validated road route: "+" → ".join(result["route_sensor_ids"]))
                    if result.get("sensor_locations"):
                        st.map(pd.DataFrame(result["sensor_locations"]).rename(columns={"latitude":"lat","longitude":"lon"}),latitude="lat",longitude="lon")
                        st.caption("Markers show the selected sensor locations only; the straight line is not a road route.")
                    st.json(result.get("predicted_sensor_speeds",{}))
                else:
                    st.metric("Estimated duration",f"{result['estimated_duration']['estimated_minutes']:.1f} min")
                    st.write("Route: "+" → ".join(result["route_sensor_ids"]))
                    st.write(f"Distance: {result['distance_km']:.2f} km")
                    st.dataframe(pd.DataFrame(result["estimated_duration"]["segments"]),use_container_width=True)
                    if result["estimated_duration"].get("extrapolated_beyond_forecast"):
                        st.warning("This trip extends past the forecast window. The last available forecast speed is reused for later segments.")
                    st.caption(f"Model {result['model_version']} · departure {result['departure_time']}")
    except Exception as exc: api_error(exc)
    st.divider()
    st.subheader("Speed forecast by sensor")
    try:
        status=get("/api/v1/dataset/status")
        if status.get("loaded"):
            selected=st.multiselect("Sensors to forecast",status["sensor_ids"][:300],default=status["sensor_ids"][:min(3,len(status["sensor_ids"]))])
            if st.button("Forecast speeds"):
                result=post("/api/v1/predictions",{"sensor_ids":selected,"horizon":horizon if 'horizon' in locals() else 1})
                rows=[{"sensor":p["sensor_id"],**point} for p in result["predictions"] for point in p["forecast"]]
                if rows: st.plotly_chart(px.line(pd.DataFrame(rows),x="horizon_step",y="predicted_speed",color="sensor",markers=True,labels={"predicted_speed":"Speed (mph)"}),use_container_width=True)
    except Exception as exc: api_error(exc)

elif page=="Dashboard Overview":
    st.header("System overview")
    try:
        x=get("/api/v1/system/status"); a,b,c,d=st.columns(4)
        a.metric("Sensors",x["dataset"].get("sensors",0)); b.metric("Observations",x["dataset"].get("rows",0)); c.metric("DCRNN ready","Yes" if x["model_ready"] else "No"); d.metric("Road distances","Available" if x["route_distance_available"] else "Missing")
        simulated=x.get("ingestion",{}).get("observations",{}).get("source_counts",{}).get("synthetic-demo",0)
        if simulated: st.warning(f"{simulated} synthetic-demo observation(s) were ingested. They are simulated, not live traffic.")
        st.json(x)
        try: st.subheader("Held-out metrics"); st.json(get("/api/v1/model/metrics"))
        except Exception: st.info("No measured test metrics yet. Train and evaluate a model first.")
        try:
            recent=get("/api/v1/travel-time/latest")["items"]
            st.subheader("Recent validated travel-time estimates")
            if recent: st.dataframe(pd.DataFrame(recent),use_container_width=True)
            else: st.caption("No persisted route estimates yet.")
        except Exception: st.caption("Travel-time history is unavailable until PostgreSQL is connected.")
    except Exception as exc: api_error(exc)

elif page=="Dataset Explorer":
    st.header("Dataset Explorer")
    uploaded=st.file_uploader("Upload traffic CSV",type=["csv"])
    if uploaded and st.button("Validate and load upload"):
        try:
            result=post("/api/v1/dataset/upload",files={"file":(uploaded.name,uploaded.getvalue(),"text/csv")}); st.success("Dataset loaded"); st.json(result)
        except Exception as exc: api_error(exc)
    if st.button("Load configured local dataset"):
        try: st.json(post("/api/v1/dataset/load",{}))
        except Exception as exc: api_error(exc)
    try:
        st.json(get("/api/v1/dataset/statistics")); preview=get("/api/v1/dataset/preview",rows=20)
        if preview["rows"]: st.dataframe(pd.DataFrame(preview["rows"]),use_container_width=True)
    except Exception as exc: api_error(exc)

elif page=="Sensor Analysis":
    st.header("Sensor analysis")
    try:
        registry=get("/api/v1/sensors",limit=1000); ids=[x["sensor_id"] for x in registry["items"]]
        chosen=st.multiselect("Select sensors",ids,default=ids[:3])
        if chosen:
            rows=[]
            for sid in chosen:
                rows.extend([{"sensor_id":sid,**x} for x in get(f"/api/v1/sensors/{sid}/history",limit=1000)["observations"]])
            series=pd.DataFrame(rows)
            if not series.empty:
                st.plotly_chart(px.line(series,x="timestamp",y="speed",color="sensor_id",labels={"speed":"Speed (mph)"}),use_container_width=True)
                st.dataframe(series.groupby("sensor_id").speed.agg(["mean","median","std","min","max"]).reset_index(),use_container_width=True)
            if len(chosen)>1:
                corr=get("/api/v1/graph/correlations",sensor_ids=chosen)
                st.plotly_chart(px.imshow(corr["correlations"],x=chosen,y=chosen,zmin=-1,zmax=1,color_continuous_scale="RdBu",text_auto=".2f"),use_container_width=True)
    except Exception as exc: api_error(exc)

elif page=="Model Training":
    st.header("DCRNN training")
    st.warning("Training uses the configured dataset, chronological splits, train-only scaling, and a train-only correlation graph.")
    with st.form("training_config"):
        c1,c2,c3=st.columns(3)
        input_window=c1.number_input("History window",min_value=1,max_value=288,value=12)
        forecast_horizon=c2.number_input("Forecast steps",min_value=1,max_value=12,value=12)
        diffusion_order=c3.number_input("Diffusion order",min_value=1,max_value=5,value=2)
        c4,c5,c6=st.columns(3)
        hidden_channels=c4.selectbox("Hidden channels",[16,32,64,128,256],index=2)
        batch_size=c5.selectbox("Batch size",[8,16,32,64,128],index=2)
        max_epochs=c6.number_input("Maximum epochs",min_value=1,max_value=1000,value=100)
        submitted=st.form_submit_button("Start training",type="primary")
    if submitted:
        try:
            payload={"input_window":input_window,"forecast_horizon":forecast_horizon,"diffusion_order":diffusion_order,"hidden_channels":hidden_channels,"batch_size":batch_size,"max_epochs":max_epochs}
            st.session_state["job"]=post("/api/v1/model/train",payload)
        except Exception as exc: api_error(exc)
    job=st.session_state.get("job")
    if job:
        try:
            latest=get(f"/api/v1/model/train/{job['job_id']}"); st.progress(latest.get("progress",0)); st.json(latest)
            if latest.get("history"):
                progress=pd.DataFrame(latest["history"]).set_index("epoch")
                st.line_chart(progress[["train_loss","validation_loss"]])
            if latest.get("status")=="running": st.button("Refresh status")
        except Exception as exc: api_error(exc)

elif page=="Model Evaluation":
    st.header("Measured test evaluation")
    try:
        m=get("/api/v1/model/metrics"); st.caption(f"Units: {m['units']}")
        c1,c2,c3=st.columns(3); c1.metric("MAE",f"{m['mae']:.3f}"); c2.metric("MSE",f"{m['mse']:.3f}"); c3.metric("RMSE",f"{m['rmse']:.3f}")
        st.plotly_chart(px.line(pd.DataFrame(m["horizon_metrics"]),x="horizon_step",y=["mae","rmse"],markers=True),use_container_width=True)
        if m.get("sensor_metrics"):
            sensor=pd.DataFrame(m["sensor_metrics"]).sort_values("mae",ascending=False)
            st.subheader("Sensor-level performance")
            st.dataframe(sensor,use_container_width=True)
        sample=get("/api/v1/model/test-sample",limit=5000)["rows"]
        if sample:
            samples=pd.DataFrame(sample)
            selected_sensor=st.selectbox("Sample sensor",samples.sensor_id.unique())
            single=samples[samples.sensor_id==selected_sensor]
            st.plotly_chart(px.scatter(single,x="actual_speed_mph",y="predicted_speed_mph",labels={"actual_speed_mph":"Actual (mph)","predicted_speed_mph":"Predicted (mph)"}),use_container_width=True)
            st.plotly_chart(px.histogram(single.assign(error=single.predicted_speed_mph-single.actual_speed_mph),x="error",labels={"error":"Prediction error (mph)"}),use_container_width=True)
        try:
            history=pd.DataFrame(get("/api/v1/model/history"))
            st.subheader("Training curves")
            st.plotly_chart(px.line(history,x="epoch",y=["train_loss","validation_loss"],markers=True),use_container_width=True)
        except Exception: pass
    except Exception as exc: api_error(exc)

elif page=="Graph & Route Diagnostics":
    st.header("Graph and route diagnostics")
    try:
        status=get("/api/v1/graph/status"); st.json(status)
        locations=get("/api/v1/graph/locations")["items"]
        if locations:
            st.subheader("Validated sensor coordinates")
            st.map(pd.DataFrame(locations).rename(columns={"latitude":"lat","longitude":"lon"}),latitude="lat",longitude="lon")
            st.caption("These are sensor markers only. Coordinates do not provide road connectivity, driving paths, or edge distances.")
        if "correlation" in str(status.get("graph_source","")):
            st.info("Correlation adjacency is used only for DCRNN message passing. It does not establish a road route or segment distance.")
        elif status.get("is_road_network"):
            st.info("The DCRNN uses supplied road edges; travel-time calculation still needs valid edge distances and speed forecasts.")
        with st.expander("Road metadata format"):
            st.code('''{"sensors":[{"sensor_id":"<dataset sensor ID>","latitude":34.0,"longitude":-118.0}],"edges":[{"source_sensor":"<ID>","destination_sensor":"<ID>","edge_distance_km":1.2,"edge_travel_time_seconds":90}]}''',language="json")
    except Exception as exc: api_error(exc)

else:
    st.header("System health and settings")
    st.write(f"API base URL: `{API_BASE_URL}`")
    try: st.json(get("/api/v1/system/status"))
    except Exception as exc: api_error(exc)

