import uvicorn

from app.main import create_app
from app.utils import g_config, setup_logging
import os
from loguru import logger

app = create_app()

if __name__ == "__main__":
    # Setup loguru logging
    setup_logging(level=g_config.logging.level)
    
    key_path = "/certs/privkey.pem"
    cert_path = "/certs/fullchain.pem"
    # Check if the certificate files exist
    if os.path.exists(key_path) and os.path.exists(cert_path):
        # If the certificates exist, run in HTTPS mode
        logger.info("Starting server in HTTPS mode...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
        )
    else:
        # Otherwise, run in HTTP mode
        logger.info("Starting server in HTTP mode (SSL certificates not found or configured)...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
        )
