import uvicorn
from lens_agent import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning", log_config=None, access_log=False)
