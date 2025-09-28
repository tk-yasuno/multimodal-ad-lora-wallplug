"""
MAD-FH: UI Launcher Script
StreamlitまたはGradio UIを起動するためのスクリプト
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def launch_streamlit():
    """Streamlit UIを起動"""
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Streamlit UIを起動中...")
    print(f"URL: http://localhost:8501")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit起動エラー: {e}")
    except KeyboardInterrupt:
        print("Streamlit UIを終了しました")


def launch_gradio():
    """Gradio UIを起動"""
    # プロジェクトルートに移動
    os.chdir(Path(__file__).parent)
    
    try:
        from src.ui.gradio_app import main
        print("🚀 Gradio UIを起動中...")
        print(f"URL: http://localhost:7860")
        main()
    except ImportError as e:
        print(f"Gradio起動エラー: {e}")
        print("必要なライブラリがインストールされていない可能性があります")
    except KeyboardInterrupt:
        print("Gradio UIを終了しました")


def main():
    parser = argparse.ArgumentParser(description='MAD-FH UI Launcher')
    parser.add_argument(
        '--ui', 
        choices=['streamlit', 'gradio'], 
        default='streamlit',
        help='UI framework to launch (default: streamlit)'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔍 MAD-FH: Multimodal Anomaly Detector")
    print("   Human-in-the-Loop UI")
    print("=" * 50)
    
    if args.ui == 'streamlit':
        launch_streamlit()
    elif args.ui == 'gradio':
        launch_gradio()


if __name__ == "__main__":
    main()
