import os

import uvicorn

from api import app


def main():
    if os.getenv("ENVIRONMENT") == "prod":
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
