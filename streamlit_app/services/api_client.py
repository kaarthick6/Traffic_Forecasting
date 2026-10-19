import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL=os.getenv("API_BASE_URL","http://localhost:8000").rstrip("/")
HEADERS={"X-Admin-Token":os.getenv("ADMIN_API_TOKEN","")} if os.getenv("ADMIN_API_TOKEN") else {}

def get(path,**params):
    with httpx.Client(timeout=30) as client:
        response=client.get(API_BASE_URL+path,params=params,headers=HEADERS); response.raise_for_status(); return response.json()

def post(path,payload=None,**kwargs):
    with httpx.Client(timeout=120) as client:
        if "files" in kwargs:
            response=client.post(API_BASE_URL+path,headers=HEADERS,**kwargs)
        else:
            response=client.post(API_BASE_URL+path,json=payload,headers=HEADERS,**kwargs)
        response.raise_for_status(); return response.json()

