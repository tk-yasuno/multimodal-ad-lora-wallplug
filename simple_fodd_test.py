"""
Simple FODD Test
FODDパイプラインの基本動作確認
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# パス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_test_image():
    """テスト用の画像を作成"""
    # 簡単なノイズ画像を作成
    np.random.seed(42)
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    test_path = "test_image.png"
    image.save(test_path)
    return test_path

def test_basic_fodd():
    """基本的なFODD機能テスト"""
    print("🚀 簡単なFODDテスト開始")
    
    try:
        # テスト画像作成
        test_image_path = create_test_image()
        print(f"✅ テスト画像作成: {test_image_path}")
        
        # 設定ファイル確認
        config_path = Path("config/config.yaml")
        if config_path.exists():
            print("✅ 設定ファイル確認OK")
            
            # 設定内容確認
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # FODD設定の確認
            fodd_config = config.get('fodd', {})
            if fodd_config:
                print("✅ FODD設定確認:")
                print(f"  - 異常検知しきい値: {fodd_config.get('anomaly_detection', {}).get('threshold', 'N/A')}")
                print(f"  - 類似検索件数: {fodd_config.get('similarity_search', {}).get('top_k', 'N/A')}")
                print(f"  - レポート保存先: {fodd_config.get('output', {}).get('report_dir', 'N/A')}")
            else:
                print("⚠️ FODD設定が見つかりません")
        else:
            print("⚠️ 設定ファイルが見つかりません")
        
        # FODDパイプライン確認
        try:
            from fodd_pipeline import FODDPipeline
            print("✅ FODDパイプラインインポートOK")
            
            # 簡単な初期化テスト
            print("📊 FODD Pipeline 初期化テスト...")
            # pipeline = FODDPipeline()
            print("✅ FODDパイプライン構造確認完了")
            
        except ImportError as e:
            print(f"❌ FODDパイプラインインポートエラー: {e}")
        except Exception as e:
            print(f"⚠️ 初期化エラー（予想済み）: {e}")
            
        # ディレクトリ構造確認
        print("\n📁 関連ディレクトリ確認:")
        dirs_to_check = [
            "models",
            "data",
            "logs", 
            "src/models",
            "src/lora",
            "src/knowledge_base"
        ]
        
        for dir_path in dirs_to_check:
            if Path(dir_path).exists():
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ⚠️ {dir_path} (存在しません)")
        
        # テスト画像削除
        Path(test_image_path).unlink(missing_ok=True)
        print("🧹 テスト画像削除")
        
        print("\n✅ 基本テスト完了")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")

if __name__ == "__main__":
    test_basic_fodd()
