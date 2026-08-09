"""Environment-based configuration for the ML Predictive Analytics platform."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.database_url = os.getenv("DATABASE_URL")
        self.model_path = os.getenv("MODEL_PATH", "./model.joblib")
        self.retrain_interval_minutes = int(os.getenv("RETRAIN_INTERVAL_MINUTES", "30"))

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


config = Config()
