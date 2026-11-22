"""
Application Entrypoint for Cloud Hosting (Hugging Face Spaces, Streamlit Cloud).
"""
import sys
import runpy
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

runpy.run_path(str(root_dir / "frontend" / "streamlit_app.py"), run_name="__main__")
