"""
FODD Pipeline 実用テスト
実際のStreamlitアプリでの動作を想定したテスト
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import os

# パス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_image_for_ui():
    """UIテスト用の画像を作成"""
    # 製造ラインの異常を想定したテスト画像
    np.random.seed(42)
    
    # ベース画像（正常パターン）
    base_image = np.zeros((224, 224, 3), dtype=np.uint8)
    base_image[:, :, 0] = 100  # 青みがかった背景
    base_image[:, :, 1] = 120
    base_image[:, :, 2] = 140
    
    # 異常パターンを追加（傷やひび割れを模倣）
    # 中央に不規則な形状
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 112, 112
    mask = ((x - center_x)**2 + (y - center_y)**2) < 20**2
    base_image[mask] = [255, 255, 255]  # 白い異常部分
    
    # ノイズ追加
    noise = np.random.normal(0, 10, (224, 224, 3))
    base_image = np.clip(base_image + noise, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(base_image)
    
    # 一時ファイルとして保存
    temp_path = "fodd_test_image.png"
    image.save(temp_path)
    return temp_path

def test_fodd_integration():
    """FODD統合機能の実用テスト"""
    print("🔧 FODD統合テスト開始")
    
    try:
        # テスト画像作成
        test_image_path = create_test_image_for_ui()
        print(f"✅ テスト画像作成: {test_image_path}")
        
        # 設定の詳細確認
        import yaml
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print("📋 FODD設定詳細:")
        fodd_config = config.get('fodd', {})
        for section, settings in fodd_config.items():
            print(f"  {section}:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {settings}")
        
        # ディレクトリ構造確認
        print("\n📁 モデル・データ構造確認:")
        important_paths = [
            "models/autoencoder_best.pth",
            "models/lora_model",
            "data/knowledge_base.json",
            "data/features.json",
            "logs"
        ]
        
        for path in important_paths:
            if Path(path).exists():
                if Path(path).is_file():
                    size = Path(path).stat().st_size / 1024 / 1024
                    print(f"  ✅ {path} ({size:.1f}MB)")
                else:
                    print(f"  ✅ {path} (ディレクトリ)")
            else:
                print(f"  ⚠️ {path} (存在しません)")
        
        # StreamlitアプリのFODD機能確認メッセージ
        print("\n🌐 Streamlit FODD機能の使用方法:")
        print("  1. ブラウザで http://localhost:8502 にアクセス")
        print("  2. サイドバーで 'FODD即時分析' を選択")
        print("  3. '単一画像分析' タブを選択")
        print(f"  4. {test_image_path} をアップロード")
        print("  5. '分析実行' ボタンをクリック")
        print("  6. 異常検知結果と説明が表示されます")
        
        print(f"\n📷 テスト画像: {test_image_path}")
        print("   - 製造ラインの異常を模擬した画像")
        print("   - 中央に白い異常部分を配置")
        print("   - 背景ノイズを追加")
        
        print("\n✅ FODD統合テスト準備完了")
        print("💡 Streamlitアプリで実際の分析をお試しください！")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fodd_integration()
