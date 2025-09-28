"""
MAD-FH: Integrated Pipeline Script
ステップ1〜3の統合実行スクリプト
"""

import argparse
import yaml
import json
from pathlib import Path
import os
import sys
from datetime import datetime

# プロジェクト内のモジュールをインポート
sys.path.append(str(Path(__file__).parent))

from src.data.metadata_manager import ImageMetadataDB, scan_and_register_images
from src.data.preprocess import create_preprocessing_pipeline
from src.training.train_autoencoder import AutoencoderTrainer
from src.training.train_simclr import SimCLRTrainer
from src.utils.logger import setup_logger


class MADFHPipeline:
    """MAD-FH統合パイプライン"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("mad_fh_pipeline")
        
        # パスの設定
        self.raw_images_path = self.config['data']['raw_images_path']
        self.processed_images_path = self.config['data']['processed_images_path']
        self.metadata_db_path = self.config['data']['metadata_db_path']
        
        # 結果保存ディレクトリ
        self.results_dir = Path("results") / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_data_management(self, camera_id: str = "cam001", location: str = "factory"):
        """
        ステップ1: 画像データの収集・管理
        
        Args:
            camera_id: カメラID
            location: 設置場所
        """
        self.logger.info("=== Step 1: Image Data Collection & Management ===")
        
        # メタデータDBの初期化
        os.makedirs(Path(self.metadata_db_path).parent, exist_ok=True)
        
        # 画像の登録
        registered_count = scan_and_register_images(
            image_directory=self.raw_images_path,
            db_path=self.metadata_db_path,
            camera_id=camera_id,
            location=location
        )
        
        # 統計情報の取得
        db = ImageMetadataDB(self.metadata_db_path)
        stats = db.get_statistics()
        
        self.logger.info(f"Registered {registered_count} new images")
        self.logger.info(f"Total images in database: {stats['total_images']}")
        
        # 結果保存
        step1_results = {
            "registered_images": registered_count,
            "database_statistics": stats,
            "camera_id": camera_id,
            "location": location
        }
        
        with open(self.results_dir / "step1_results.json", 'w', encoding='utf-8') as f:
            json.dump(step1_results, f, indent=2, ensure_ascii=False)
        
        return step1_results
    
    def step2_preprocessing(self, sample_size: int = None):
        """
        ステップ2: 画像前処理とサンプリング
        
        Args:
            sample_size: サンプル数（Noneの場合は全データ）
        """
        self.logger.info("=== Step 2: Image Preprocessing & Sampling ===")
        
        # 前処理パイプラインの実行
        preprocessing_results = create_preprocessing_pipeline(
            config_path=self.config_path,
            db_path=self.metadata_db_path,
            output_dir=self.processed_images_path,
            sample_size=sample_size
        )
        
        self.logger.info(f"Processed {preprocessing_results['processed_count']} images")
        
        # 結果保存
        with open(self.results_dir / "step2_results.json", 'w', encoding='utf-8') as f:
            json.dump(preprocessing_results, f, indent=2, ensure_ascii=False)
        
        return preprocessing_results
    
    def step3_model_training(self, model_types: list = ["autoencoder"]):
        """
        ステップ3: 異常検知モデルの構築
        
        Args:
            model_types: 学習するモデルタイプのリスト
        """
        self.logger.info("=== Step 3: Anomaly Detection Model Training ===")
        
        training_results = {}
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type} model...")
            
            if model_type in ["autoencoder", "vae"]:
                # Autoencoder系の学習
                trainer = AutoencoderTrainer(self.config_path)
                model = trainer.train(self.processed_images_path, model_type)
                
                training_results[model_type] = {
                    "best_val_loss": trainer.best_val_loss,
                    "total_epochs": len(trainer.train_losses),
                    "results_dir": str(trainer.results_dir)
                }
                
            elif model_type == "simclr":
                # SimCLRの学習
                trainer = SimCLRTrainer(self.config_path)
                model = trainer.train(self.raw_images_path)
                
                training_results[model_type] = {
                    "best_loss": trainer.best_loss,
                    "total_epochs": len(trainer.train_losses),
                    "results_dir": str(trainer.results_dir)
                }
            
            else:
                self.logger.warning(f"Unknown model type: {model_type}")
        
        # 結果保存
        with open(self.results_dir / "step3_results.json", 'w', encoding='utf-8') as f:
            json.dump(training_results, f, indent=2, ensure_ascii=False)
        
        return training_results
    
    def run_full_pipeline(self, 
                         camera_id: str = "cam001",
                         location: str = "factory",
                         sample_size: int = None,
                         model_types: list = ["autoencoder"]):
        """
        フルパイプラインの実行
        
        Args:
            camera_id: カメラID
            location: 設置場所
            sample_size: サンプル数
            model_types: 学習するモデルタイプ
        """
        self.logger.info("=== MAD-FH Pipeline Started ===")
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "parameters": {
                "camera_id": camera_id,
                "location": location,
                "sample_size": sample_size,
                "model_types": model_types
            }
        }
        
        try:
            # ステップ1: データ管理
            step1_results = self.step1_data_management(camera_id, location)
            pipeline_results["step1"] = step1_results
            
            # ステップ2: 前処理
            step2_results = self.step2_preprocessing(sample_size)
            pipeline_results["step2"] = step2_results
            
            # ステップ3: モデル学習
            step3_results = self.step3_model_training(model_types)
            pipeline_results["step3"] = step3_results
            
            pipeline_results["status"] = "completed"
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            self.logger.info("=== MAD-FH Pipeline Completed Successfully ===")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # 最終結果の保存
            with open(self.results_dir / "pipeline_results.json", 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
        
        return pipeline_results


def create_sample_images(output_dir: str, num_images: int = 50):
    """
    テスト用のサンプル画像を生成
    
    Args:
        output_dir: 出力ディレクトリ
        num_images: 生成する画像数
    """
    import numpy as np
    from PIL import Image
    from datetime import datetime, timedelta
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} sample images in {output_dir}")
    
    # ベースとなる日時
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(num_images):
        # ランダムな画像を生成（1920x1080）
        # 実際の工場画像を模倣した内容
        image_array = np.random.randint(50, 200, size=(1080, 1920, 3), dtype=np.uint8)
        
        # 中央に矩形を追加（機械や設備を模倣）
        h, w = 1080, 1920
        center_h, center_w = h // 2, w // 2
        rect_h, rect_w = 200, 300
        
        image_array[
            center_h - rect_h//2:center_h + rect_h//2,
            center_w - rect_w//2:center_w + rect_w//2
        ] = [100, 150, 100]  # 緑っぽい色
        
        # ファイル名生成（YYYYMMDD_HHMMSS_cameraID.jpg）
        timestamp = base_time + timedelta(hours=i*2)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_cam001.jpg"
        
        # 画像保存
        image = Image.fromarray(image_array)
        image.save(output_path / filename, quality=90)
    
    print(f"Generated {num_images} sample images successfully")


def main():
    parser = argparse.ArgumentParser(description='MAD-FH Pipeline Runner')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--camera_id', type=str, default='cam001', help='Camera ID')
    parser.add_argument('--location', type=str, default='factory', help='Installation location')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for preprocessing')
    parser.add_argument('--model_types', nargs='+', default=['autoencoder'], choices=['autoencoder', 'vae', 'simclr'], help='Model types to train')
    parser.add_argument('--generate_samples', action='store_true', help='Generate sample images')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of sample images to generate')
    
    args = parser.parse_args()
    
    # サンプル画像生成（指定された場合）
    if args.generate_samples:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        raw_images_path = config['data']['raw_images_path']
        create_sample_images(raw_images_path, args.num_samples)
    
    # パイプラインの実行
    pipeline = MADFHPipeline(args.config)
    
    results = pipeline.run_full_pipeline(
        camera_id=args.camera_id,
        location=args.location,
        sample_size=args.sample_size,
        model_types=args.model_types
    )
    
    print(f"\n=== Pipeline Results ===")
    print(f"Status: {results['status']}")
    print(f"Results saved to: {pipeline.results_dir}")
    
    if results['status'] == 'completed':
        print(f"Step 1 - Registered images: {results['step1']['registered_images']}")
        print(f"Step 2 - Processed images: {results['step2']['processed_count']}")
        print(f"Step 3 - Trained models: {list(results['step3'].keys())}")


if __name__ == "__main__":
    main()
