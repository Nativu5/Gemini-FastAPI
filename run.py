import uvicorn

from app.main import create_app
from app.utils import g_config, setup_logging

app = create_app()

if __name__ == "__main__":
    # Setup loguru logging
    setup_logging(level=g_config.logging.level)

    uvicorn.run(
        app,
        host=g_config.server.host,
        port=g_config.server.port,
        log_config=None,
    )
