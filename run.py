import os
import sys

import uvicorn
from loguru import logger

from app.main import create_app
from app.utils import g_config, setup_logging

app = create_app()

if __name__ == "__main__":
    setup_logging(level=g_config.logging.level)

    server, https = g_config.server, g_config.server.https
    if https.enabled and not (os.path.exists(https.key_file) and os.path.exists(https.cert_file)):
        logger.critical(
            f"HTTPS enabled but SSL certificate files not found: {https.key_file}, {https.cert_file}"
        )
        sys.exit(1)

    scheme = "https" if https.enabled else "http"
    logger.info(f"Starting server at {scheme}://{server.host}:{server.port} ...")
    uvicorn.run(
        app,
        host=server.host,
        port=server.port,
        log_config=None,
        ssl_keyfile=https.key_file if https.enabled else None,
        ssl_certfile=https.cert_file if https.enabled else None,
    )
