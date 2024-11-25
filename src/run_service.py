import warnings
from pydantic import warnings as pydantic_warnings
import uvicorn
from intent_service.api import app

warnings.filterwarnings("ignore", category=pydantic_warnings.PydanticDeprecatedSince20)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
