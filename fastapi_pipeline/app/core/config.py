import os

def get_settings():
    env = os.getenv("ENV", "local")

    if env == "docker":
        from .settings.docker import DockerConfig
        return DockerConfig()
    elif env == "prod":
        pass
    else:
        from .settings.local import LocalConfig
        return LocalConfig()

settings = get_settings()
print(f'settings: {settings}')
