from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import get_settings

class Base(DeclarativeBase): pass

def make_engine(url: str | None = None):
    resolved=url or get_settings().sqlalchemy_url
    options={"pool_pre_ping":True,"future":True}
    if resolved.startswith("postgresql"): options["connect_args"]={"connect_timeout":3}
    return create_engine(resolved,**options)

engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

