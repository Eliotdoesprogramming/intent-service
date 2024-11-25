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
    elif os.getenv("VSCODE_DEBUGGER"):
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000        )
    else:
        uvicorn.run(
            "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
