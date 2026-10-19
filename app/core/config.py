from functools import lru_cache
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    dataset_path: str = r"C:\Users\Karthick\Downloads\METR-LA (1).csv"
    dataset_timestamp_column: str | None = None
    dataset_sensor_id_column: str | None = None
    dataset_value_column: str | None = None
    sensor_metadata_path: str | None = None
    artifact_dir: Path = Path("artifacts")
    database_url: str | None = None
    admin_api_token: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "traffic_prediction"
    db_username: str = "postgres"
    db_password: str = "password"
    input_window: int = 12
    forecast_horizon: int = 12
    diffusion_order: int = 2
    hidden_channels: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    graph_top_k: int = 8
    graph_correlation_threshold: float = 0.3
    zero_is_missing: bool = True
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "traffic-observations"
    kafka_consumer_group: str = "dcrnn-traffic-api"

    @model_validator(mode="after")
    def validate_training_settings(self):
        if min(self.input_window,self.forecast_horizon,self.diffusion_order,self.hidden_channels,self.encoder_layers,self.decoder_layers,self.batch_size,self.max_epochs,self.early_stopping_patience,self.graph_top_k) < 1:
            raise ValueError("Window, horizon, model dimensions, batch size, training limits, patience, and graph top-k must be positive.")
        if not 0 <= self.graph_correlation_threshold <= 1:
            raise ValueError("GRAPH_CORRELATION_THRESHOLD must be between 0 and 1.")
        if self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE must be positive.")
        return self

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url or (
            f"postgresql+psycopg://{self.db_username}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

