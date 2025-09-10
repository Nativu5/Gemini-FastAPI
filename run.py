import os

import uvicorn
from loguru import logger

from app.main import create_app
from app.utils import g_config, setup_logging

app = create_app()

if __name__ == "__main__":
    # Setup loguru logging
    setup_logging(level=g_config.logging.level)

    # Check HTTPS configuration
    if g_config.server.https.enabled:
        key_path = g_config.server.https.key_file
        cert_path = g_config.server.https.cert_file

        # Check if the certificate files exist
        if not os.path.exists(key_path) or not os.path.exists(cert_path):
            logger.critical(f"HTTPS enabled but SSL certificate files not found: {key_path}, {cert_path}")
            exit(1)

        logger.info(f"Starting server at https://{g_config.server.host}:{g_config.server.port} ...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
        )
    else:
        logger.info(f"Starting server at http://{g_config.server.host}:{g_config.server.port} ...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
        )
