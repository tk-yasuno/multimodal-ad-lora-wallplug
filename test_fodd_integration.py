"""
FODD Pipeline テストスクリプト
ステップ7の動作確認用
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_fodd_pipeline():
    """FODD Pipelineの基本テスト"""
    print("🚀 FODD Pipeline テスト開始")
    
    try:
        from fodd_pipeline import FODDPipeline
        
        # Pipeline初期化
        print("📊 FODD Pipeline 初期化中...")
        pipeline = FODDPipeline()
        print("✅ 初期化完了")
        
        # テスト用サンプル画像作成
        test_image_path = create_test_image()
        print(f"🖼️ テスト画像作成: {test_image_path}")
        
        # 単一画像処理テスト
        print("🔍 単一画像処理テスト実行中...")
        result = pipeline.process_single_image(test_image_path)
        
        # 結果表示
        print("\n📋 分析結果:")
        print(f"  - 処理時間: {result.get('processing_time', 0):.2f}秒")
        print(f"  - 異常判定: {'異常' if result.get('anomaly_detection', {}).get('is_anomaly', False) else '正常'}")
        print(f"  - 異常スコア: {result.get('anomaly_detection', {}).get('anomaly_score', 0):.3f}")
        print(f"  - 類似事例数: {len(result.get('similar_cases', []))}")
        print(f"  - 生成説明: {result.get('generated_description', 'N/A')}")
        
        if 'report_path' in result:
            print(f"  - レポート保存: {result['report_path']}")
        
        # テスト画像削除
        Path(test_image_path).unlink(missing_ok=True)
        
        print("✅ FODD Pipeline テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ FODD Pipeline テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """テスト用画像を作成"""
    # 512x512のテスト画像（グラデーション + ノイズ）
    width, height = 512, 512
    
    # グラデーション作成
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 円形パターン
    center_x, center_y = 0.5, 0.5
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    pattern = np.sin(radius * 10) * 0.5 + 0.5
    
    # ノイズ追加
    noise = np.random.random((height, width)) * 0.2
    
    # RGB画像作成
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # チャンネル別設定
    image_array[:, :, 0] = ((pattern + noise) * 255).astype(np.uint8)  # Red
    image_array[:, :, 1] = ((pattern * 0.8 + noise) * 255).astype(np.uint8)  # Green
    image_array[:, :, 2] = ((pattern * 0.6 + noise) * 255).astype(np.uint8)  # Blue
    
    # 異常パターン追加（ランダム位置に明るいスポット）
    for _ in range(3):
        spot_x = np.random.randint(50, width-50)
        spot_y = np.random.randint(50, height-50)
        spot_size = np.random.randint(10, 30)
        
        y_start, y_end = max(0, spot_y-spot_size), min(height, spot_y+spot_size)
        x_start, x_end = max(0, spot_x-spot_size), min(width, spot_x+spot_size)
        
        image_array[y_start:y_end, x_start:x_end, :] = 255
    
    # PIL Imageに変換
    image = Image.fromarray(image_array)
    
    # 保存
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path = test_dir / "fodd_test_image.jpg"
    image.save(test_path)
    
    return str(test_path)

def test_streamlit_integration():
    """Streamlit統合テスト"""
    print("\n🌐 Streamlit統合テスト")
    
    try:
        # Streamlitアプリのインポートテスト
        from src.ui.streamlit_app import MADFHApp
        print("✅ Streamlitアプリインポート成功")
        
        # 基本初期化テスト
        # 注意: 実際のStreamlit環境外では一部機能が制限される
        print("📱 アプリ初期化テスト（制限モード）")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit統合テストエラー: {e}")
        return False

def test_knowledge_base_integration():
    """Knowledge Base統合テスト"""
    print("\n🧠 Knowledge Base統合テスト")
    
    try:
        from src.knowledge_base.knowledge_manager import KnowledgeBaseManager
        import yaml
        
        # 設定読み込み
        config_path = "config/config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Knowledge Base Manager
        kb_manager = KnowledgeBaseManager(config)
        print("✅ Knowledge Base Manager初期化成功")
        
        # 統計情報取得テスト
        stats = kb_manager.get_knowledge_base_stats()
        print(f"📊 Knowledge Base統計: {stats.get('total_features', 0)}件の特徴量")
        
        # 検索テスト
        search_results = kb_manager.search_similar_features("テスト検索", max_results=3)
        print(f"🔍 検索テスト: {len(search_results)}件の結果")
        
        kb_manager.close()
        print("✅ Knowledge Base統合テスト完了")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Base統合テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=" * 50)
    print("MAD-FH ステップ7 (FODD) 統合テスト")
    print("=" * 50)
    
    test_results = {}
    
    # 各テスト実行
    test_results['knowledge_base'] = test_knowledge_base_integration()
    test_results['fodd_pipeline'] = test_fodd_pipeline()
    test_results['streamlit_integration'] = test_streamlit_integration()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name:25}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n合計: {passed_tests}/{total_tests} テスト成功")
    
    if passed_tests == total_tests:
        print("🎉 すべてのテストが成功しました！")
        print("📱 Streamlitアプリでの動作確認を推奨します:")
        print("   streamlit run src/ui/streamlit_app.py --server.port 8502")
    else:
        print("⚠️  一部のテストが失敗しました。エラーを確認してください。")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
