"""
MVTec AD Wallplugs データセット実用テスト
前処理済みwallplugsデータをMAD-FH FODDパイプラインで実際にテスト
"""

import sys
from pathlib import Path
import json
import random
from datetime import datetime

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_fodd_with_wallplugs():
    """FODDパイプラインでwallplugsデータをテスト"""
    print("MVTec AD Wallplugs x FODD パイプライン テスト")
    
    # データディレクトリ
    data_dir = Path("data/processed/wallplugs")
    
    if not data_dir.exists():
        print("[ERROR] 前処理済みデータが見つかりません。preprocess_mvtec.pyを先に実行してください。")
        return
    
    # サンプル画像選択
    categories = [
        ("train/normal", "正常訓練"),
        ("train/anomalous", "異常訓練"),
        ("validation/normal", "正常検証"),
        ("validation/anomalous", "異常検証")
    ]
    
    test_results = {
        "dataset": "wallplugs",
        "test_timestamp": datetime.now().isoformat(),
        "test_results": {}
    }
    
    try:
        # FODD Pipeline初期化
        from fodd_pipeline import FODDPipeline
        pipeline = FODDPipeline()
        print("[OK] FODD Pipeline初期化完了")
        
        for category_path, category_name in categories:
            full_path = data_dir / category_path
            if not full_path.exists():
                continue
                
            # ランダムサンプル選択（2枚）
            image_files = list(full_path.glob("*.png"))
            sample_files = random.sample(image_files, min(2, len(image_files)))
            
            print(f"\n📁 {category_name} テスト ({len(sample_files)}枚)")
            category_results = []
            
            for img_file in sample_files:
                try:
                    print(f"  🖼️  {img_file.name} 分析中...")
                    
                    # FODD分析実行
                    result = pipeline.process_single_image(str(img_file))
                    
                    # 結果表示
                    anomaly_info = result.get("anomaly_detection", {})
                    is_anomaly = anomaly_info.get("is_anomaly", False)
                    anomaly_score = anomaly_info.get("score", 0.0)
                    confidence = anomaly_info.get("confidence", 0.0)
                    
                    similar_count = len(result.get("similar_cases", []))
                    description = result.get("generated_description", "")
                    processing_time = result.get("processing_time", 0.0)
                    
                    print(f"     異常判定: {'🔴 異常' if is_anomaly else '🟢 正常'}")
                    print(f"     異常スコア: {anomaly_score:.3f}")
                    print(f"     信頼度: {confidence:.3f}")
                    print(f"     類似事例: {similar_count}件")
                    print(f"     処理時間: {processing_time:.2f}秒")
                    if description:
                        print(f"     AI説明: {description[:100]}...")
                    
                    # 結果保存
                    category_results.append({
                        "image_file": img_file.name,
                        "anomaly_detection": anomaly_info,
                        "similar_cases_count": similar_count,
                        "description_length": len(description),
                        "processing_time": processing_time,
                        "success": True
                    })
                    
                except Exception as e:
                    print(f"     [ERROR] エラー: {e}")
                    category_results.append({
                        "image_file": img_file.name,
                        "error": str(e),
                        "success": False
                    })
            
            test_results["test_results"][category_name] = category_results
        
        # テスト結果サマリー
        print("\n" + "="*60)
        print("🎯 テスト結果サマリー")
        print("="*60)
        
        total_tests = 0
        successful_tests = 0
        total_anomaly_detected = 0
        total_normal_detected = 0
        
        for category_name, results in test_results["test_results"].items():
            success_count = sum(1 for r in results if r.get("success", False))
            total_count = len(results)
            
            anomaly_count = sum(1 for r in results 
                              if r.get("success") and r.get("anomaly_detection", {}).get("is_anomaly", False))
            
            print(f"{category_name}:")
            print(f"  テスト数: {total_count}枚")
            print(f"  成功: {success_count}枚")
            print(f"  異常検知: {anomaly_count}枚")
            
            total_tests += total_count
            successful_tests += success_count
            
            if "異常" in category_name:
                total_anomaly_detected += anomaly_count
            else:
                total_normal_detected += (success_count - anomaly_count)
        
        success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
        print(f"\n[STATS] 全体統計:")
        print(f"  総テスト数: {total_tests}枚")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  異常データでの異常検知: {total_anomaly_detected}件")
        print(f"  正常データでの正常判定: {total_normal_detected}件")
        
        # 結果保存
        results_file = data_dir / "fodd_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 詳細結果保存: {results_file}")
        print("="*60)
        
        # 次のステップ提案
        print("\n[NEXT] 次のステップ:")
        print("1. 異常検知性能の詳細評価")
        print("2. モデルの追加学習（必要に応じて）")
        print("3. 他のMVTec ADデータセットでのテスト")
        print("4. 製品環境での運用テスト")
        
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

def create_sample_analysis():
    """サンプル分析用の便利関数"""
    print("\n[SAMPLE] 個別サンプル分析例")
    
    # 特定の画像を指定して分析
    data_dir = Path("data/processed/wallplugs")
    sample_categories = [
        "train/normal",
        "train/anomalous"
    ]
    
    for category in sample_categories:
        category_path = data_dir / category
        if category_path.exists():
            image_files = list(category_path.glob("*.png"))[:1]  # 1枚だけ
            
            for img_file in image_files:
                print(f"\n[IMAGE] 分析対象: {category}/{img_file.name}")
                print(f"   パス: {img_file}")
                print("   コマンド例:")
                print(f"   python -c \"from fodd_pipeline import FODDPipeline; p=FODDPipeline(); print(p.process_single_image('{img_file}'))\"")

if __name__ == "__main__":
    # メイン テスト実行
    test_fodd_with_wallplugs()
    
    # 個別分析例表示
    create_sample_analysis()