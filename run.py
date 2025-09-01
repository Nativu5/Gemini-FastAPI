import uvicorn

from app.main import create_app
from app.utils import g_config, setup_logging

app = create_app()

if __name__ == "__main__":
    # Setup loguru logging
    setup_logging(level=g_config.logging.level)


    key_path = "/certs/privkey.pem"
    cert_path = "/certs/fullchain.pem"
    # 检查证书文件是否存在且配置了路径
    if os.path.exists(key_path) and os.path.exists(cert_path):
        # 如果证书存在，则以 HTTPS 模式运行
        print("Starting server in HTTPS mode...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
        )
    else:
        # 否则，以 HTTP 模式运行
        print("Starting server in HTTP mode (SSL certificates not found or configured)...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
        )
