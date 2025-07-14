from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    DEBUG: bool = False
    TRITON_URL: str
    REDIS_URL: str
    USE_BATCH_INFER : bool = True

    class Config:
        env_file_encoding = "utf-8"
        extra = "forbid"  # undefined env var 에러 방지용
