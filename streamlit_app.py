"""
Streamlit Cloud Entrypoint for Smart Document Intelligence.
Redirects execution to frontend/streamlit_app.py.
"""
import sys
import runpy
from pathlib import Path

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute the main Streamlit application
runpy.run_path(str(root_dir / "frontend" / "streamlit_app.py"), run_name="__main__")
