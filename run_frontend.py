"""
Streamlit Frontend Launcher.
Runs the Smart Document Intelligence web interface.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    app_file = project_root / "frontend" / "streamlit_app.py"
    
    print("=" * 60)
    print("Launching Smart Document Intelligence Streamlit Frontend...")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.port=8501",
        "--server.headless=true",
        "--theme.base=light"
    ]
    subprocess.run(cmd)
