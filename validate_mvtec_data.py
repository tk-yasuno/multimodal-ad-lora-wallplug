"""
MVTec AD Wallplugs データセット検証スクリプト
前処理済みデータの品質確認とMAD-FHシステムでの使用可能性テスト
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class MVTecDataValidator:
    """MVTec AD前処理済みデータの検証クラス"""
    
    def __init__(self, dataset_name="wallplugs"):
        self.dataset_name = dataset_name
        self.data_dir = Path(f"data/processed/{dataset_name}")
        self.stats_file = self.data_dir / "preprocessing_stats.json"
        
        # 統計情報読み込み
        if self.stats_file.exists():
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.preprocessing_stats = json.load(f)
        else:
            self.preprocessing_stats = None
    
    def verify_data_structure(self):
        """データ構造の確認"""
        print(f"🔍 {self.dataset_name} データ構造確認")
        
        required_dirs = [
            "train/normal",
            "train/anomalous", 
            "validation/normal",
            "validation/anomalous"
        ]
        
        structure_ok = True
        for dir_path in required_dirs:
            full_path = self.data_dir / dir_path
            if full_path.exists():
                count = len(list(full_path.glob("*.png")))
                print(f"  ✅ {dir_path}: {count}枚")
            else:
                print(f"  ❌ {dir_path}: 存在しません")
                structure_ok = False
        
        return structure_ok
    
    def verify_image_quality(self, sample_count=5):
        """画像品質の確認"""
        print(f"\n🖼️  画像品質確認（サンプル{sample_count}枚）")
        
        # 各カテゴリから サンプル取得
        categories = [
            ("train/normal", "正常訓練"),
            ("train/anomalous", "異常訓練"),
            ("validation/normal", "正常検証"),
            ("validation/anomalous", "異常検証")
        ]
        
        quality_results = {}
        
        for category_path, category_name in categories:
            full_path = self.data_dir / category_path
            if not full_path.exists():
                continue
                
            image_files = list(full_path.glob("*.png"))[:sample_count]
            
            sizes = []
            modes = []
            file_sizes = []
            
            print(f"\n  📁 {category_name} ({len(image_files)}サンプル):")
            
            for img_file in image_files:
                try:
                    # 画像情報取得
                    with Image.open(img_file) as img:
                        sizes.append(img.size)
                        modes.append(img.mode)
                    
                    # ファイルサイズ
                    file_size_kb = img_file.stat().st_size / 1024
                    file_sizes.append(file_size_kb)
                    
                    print(f"    ✅ {img_file.name}: {img.size}, {img.mode}, {file_size_kb:.1f}KB")
                    
                except Exception as e:
                    print(f"    ❌ {img_file.name}: エラー {e}")
            
            # 統計情報
            if sizes:
                unique_sizes = list(set(sizes))
                unique_modes = list(set(modes))
                avg_file_size = np.mean(file_sizes)
                
                quality_results[category_name] = {
                    "sample_count": len(image_files),
                    "sizes": unique_sizes,
                    "modes": unique_modes,
                    "avg_file_size_kb": avg_file_size
                }
                
                print(f"    📊 サイズ: {unique_sizes}")
                print(f"    📊 モード: {unique_modes}")
                print(f"    📊 平均ファイルサイズ: {avg_file_size:.1f}KB")
        
        return quality_results
    
    def create_sample_visualization(self, samples_per_category=2):
        """サンプル画像の可視化"""
        print(f"\n🎨 サンプル画像可視化（各カテゴリ{samples_per_category}枚）")
        
        categories = [
            ("train/normal", "Train Normal"),
            ("train/anomalous", "Train Anomalous"),
            ("validation/normal", "Val Normal"),
            ("validation/anomalous", "Val Anomalous")
        ]
        
        fig, axes = plt.subplots(len(categories), samples_per_category, 
                                figsize=(samples_per_category*4, len(categories)*3))
        
        if len(categories) == 1:
            axes = axes.reshape(1, -1)
        if samples_per_category == 1:
            axes = axes.reshape(-1, 1)
        
        visualization_created = False
        
        for row, (category_path, category_name) in enumerate(categories):
            full_path = self.data_dir / category_path
            
            if full_path.exists():
                image_files = list(full_path.glob("*.png"))[:samples_per_category]
                
                for col, img_file in enumerate(image_files):
                    try:
                        with Image.open(img_file) as img:
                            # RGB変換
                            if img.mode != 'RGB':
                                img_rgb = img.convert('RGB')
                            else:
                                img_rgb = img
                            
                            axes[row, col].imshow(np.array(img_rgb))
                            axes[row, col].set_title(f"{category_name}\n{img_file.name}", 
                                                   fontsize=10)
                            axes[row, col].axis('off')
                            visualization_created = True
                    
                    except Exception as e:
                        axes[row, col].text(0.5, 0.5, f"Error:\n{str(e)}", 
                                          ha='center', va='center')
                        axes[row, col].axis('off')
                
                # 不足分は空白
                for col in range(len(image_files), samples_per_category):
                    axes[row, col].axis('off')
        
        if visualization_created:
            plt.tight_layout()
            output_path = self.data_dir / f"{self.dataset_name}_samples.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ サンプル画像保存: {output_path}")
        else:
            plt.close()
            print(f"  ⚠️ サンプル画像作成に失敗")
        
        return visualization_created
    
    def test_mad_fh_integration(self):
        """MAD-FHシステムとの統合テスト"""
        print(f"\n🔧 MAD-FH統合テスト")
        
        integration_results = {
            "preprocessor_compatible": False,
            "autoencoder_compatible": False,
            "fodd_compatible": False,
            "test_results": {}
        }
        
        try:
            # 1. データ前処理互換性テスト
            from src.data.preprocess import ImagePreprocessor
            
            # 設定ファイル読み込み
            config_path = Path("config/config.yaml")
            if config_path.exists():
                preprocessor = ImagePreprocessor(str(config_path))
                
                # サンプル画像でテスト
                sample_image = next(iter((self.data_dir / "train/normal").glob("*.png")))
                processed = preprocessor.preprocess_single_image(str(sample_image))
                
                integration_results["preprocessor_compatible"] = True
                integration_results["test_results"]["preprocessor"] = {
                    "input_shape": processed.shape if hasattr(processed, 'shape') else str(type(processed)),
                    "sample_file": sample_image.name
                }
                print(f"  ✅ データ前処理: 互換性OK")
                print(f"     入力形状: {processed.shape if hasattr(processed, 'shape') else type(processed)}")
            
        except Exception as e:
            print(f"  ❌ データ前処理: エラー {e}")
            integration_results["test_results"]["preprocessor_error"] = str(e)
        
        try:
            # 2. FODD Pipeline互換性テスト
            from fodd_pipeline import FODDPipeline
            
            # FODD Pipeline初期化テスト（実際の処理はしない）
            pipeline = FODDPipeline()
            integration_results["fodd_compatible"] = True
            print(f"  ✅ FODD Pipeline: 初期化OK")
            
        except Exception as e:
            print(f"  ⚠️ FODD Pipeline: エラー {e}")
            integration_results["test_results"]["fodd_error"] = str(e)
        
        return integration_results
    
    def generate_validation_report(self):
        """検証レポート生成"""
        print(f"\n📊 検証レポート生成")
        
        # データ構造確認
        structure_ok = self.verify_data_structure()
        
        # 画像品質確認
        quality_results = self.verify_image_quality()
        
        # 可視化作成
        viz_created = self.create_sample_visualization()
        
        # MAD-FH統合テスト
        integration_results = self.test_mad_fh_integration()
        
        # レポート作成
        report = {
            "dataset_name": self.dataset_name,
            "validation_timestamp": datetime.now().isoformat(),
            "data_structure": {
                "valid": structure_ok,
            },
            "image_quality": quality_results,
            "visualization_created": viz_created,
            "mad_fh_integration": integration_results,
            "preprocessing_stats": self.preprocessing_stats
        }
        
        # レポート保存
        report_file = self.data_dir / f"{self.dataset_name}_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 検証レポート保存: {report_file}")
        
        return report
    
    def print_summary(self, report):
        """検証結果サマリー表示"""
        print("\n" + "="*60)
        print(f"🎯 MVTec AD {self.dataset_name} 検証完了")
        print("="*60)
        
        if self.preprocessing_stats:
            stats = self.preprocessing_stats["dataset_info"]
            print(f"📊 データセット情報:")
            print(f"   総画像数: {stats['processed_total']}枚")
            print(f"   訓練用正常: {stats['train_normal']}枚")
            print(f"   訓練用異常: {stats['train_anomalous']}枚") 
            print(f"   検証用正常: {stats['val_normal']}枚")
            print(f"   検証用異常: {stats['val_anomalous']}枚")
            print(f"   画像サイズ: {stats['target_size'][0]}×{stats['target_size'][1]}")
        
        print(f"\n✅ 検証結果:")
        print(f"   データ構造: {'OK' if report['data_structure']['valid'] else 'NG'}")
        print(f"   画像品質: {'OK' if report['image_quality'] else 'NG'}")
        print(f"   サンプル可視化: {'OK' if report['visualization_created'] else 'NG'}")
        
        integration = report['mad_fh_integration']
        print(f"   MAD-FH統合:")
        print(f"     前処理: {'OK' if integration['preprocessor_compatible'] else 'NG'}")
        print(f"     FODD Pipeline: {'OK' if integration['fodd_compatible'] else 'NG'}")
        
        print(f"\n📁 出力ディレクトリ: {self.data_dir}")
        print("="*60)

def main():
    """メイン実行"""
    print("🚀 MVTec AD Wallplugs データセット検証")
    print("前処理済みデータの品質確認とMAD-FHシステム統合テスト")
    
    # 検証実行
    validator = MVTecDataValidator("wallplugs")
    report = validator.generate_validation_report()
    validator.print_summary(report)
    
    print("\n💡 次のステップ:")
    print("1. MAD-FHシステムでの異常検知モデル学習")
    print("2. FODDパイプラインでの実際の分析テスト")
    print("3. 他のデータセット（sheet_metal, wallnuts, fruit_jelly）の前処理")

if __name__ == "__main__":
    main()