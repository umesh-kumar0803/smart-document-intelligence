"""
FastAPI Server Launcher.
Starts Uvicorn server hosting the Smart Document Intelligence API.
"""
import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import HOST, PORT

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Smart Document Intelligence FastAPI Server")
    print(f"API Documentation available at: http://127.0.0.1:{PORT}/docs")
    print("=" * 60)
    uvicorn.run("app.main_api:app", host="127.0.0.1", port=PORT, reload=False)
