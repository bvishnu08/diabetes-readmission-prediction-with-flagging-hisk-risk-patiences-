#!/usr/bin/env python
"""
Script to run the Streamlit dashboard
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()

