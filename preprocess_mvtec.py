"""
MVTec AD Dataset Preprocessing Script
MVTec Anomaly Detection データセット用の前処理スクリプト

対象データセット: wallplugs
- Normal train: 293枚
- Normal validation: 33枚  
- Anomalous: 90枚
- 元サイズ: 2448 x 2048
- 目標サイズ: 1024 x 1024
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class MVTecPreprocessor:
    """MVTec AD データセット前処理クラス"""
    
    def __init__(self, dataset_name="wallplugs", target_size=(1024, 1024)):
        self.dataset_name = dataset_name
        self.target_size = target_size
        self.source_dir = Path(f"data/images/{dataset_name}/TrainVald")
        self.output_dir = Path(f"data/processed/{dataset_name}")
        
        # ログ設定
        self.logger = logging.getLogger(f"mvtec_preprocessor_{dataset_name}")
        self.logger.setLevel(logging.INFO)
        
        # 統計情報
        self.stats = {
            "processed_count": 0,
            "original_sizes": [],
            "processing_errors": [],
            "dataset_info": {}
        }
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "normal").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "anomalous").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "validation" / "normal").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "validation" / "anomalous").mkdir(parents=True, exist_ok=True)
        
    def get_image_info(self, image_path):
        """画像の基本情報を取得"""
        try:
            with Image.open(image_path) as img:
                return {
                    "size": img.size,  # (width, height)
                    "mode": img.mode,
                    "format": img.format
                }
        except Exception as e:
            self.logger.error(f"画像情報取得エラー {image_path}: {e}")
            return None
    
    def resize_image(self, image_path, output_path):
        """画像をリサイズして保存"""
        try:
            with Image.open(image_path) as img:
                # 元サイズを記録
                original_size = img.size
                self.stats["original_sizes"].append(original_size)
                
                # アスペクト比を保持してリサイズ
                img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                # RGB変換（必要に応じて）
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                
                # 保存
                img_resized.save(output_path, 'PNG', optimize=True)
                
                return True, original_size
                
        except Exception as e:
            error_msg = f"画像処理エラー {image_path}: {e}"
            self.logger.error(error_msg)
            self.stats["processing_errors"].append(error_msg)
            return False, None
    
    def process_directory(self, source_subdir, output_subdir, category="normal"):
        """指定ディレクトリの全画像を処理"""
        source_path = self.source_dir / source_subdir
        output_path = self.output_dir / output_subdir
        
        if not source_path.exists():
            self.logger.warning(f"ソースディレクトリが存在しません: {source_path}")
            return 0
        
        # PNG ファイルを取得
        image_files = list(source_path.glob("*.png"))
        processed_count = 0
        
        print(f"\n📁 処理中: {source_subdir} → {output_subdir}")
        print(f"   ファイル数: {len(image_files)}枚")
        
        for image_file in tqdm(image_files, desc=f"処理中 {category}"):
            output_file = output_path / image_file.name
            success, original_size = self.resize_image(image_file, output_file)
            
            if success:
                processed_count += 1
                self.stats["processed_count"] += 1
            
        return processed_count
    
    def analyze_dataset(self):
        """データセットの分析"""
        print(f"\n🔍 {self.dataset_name} データセット分析開始")
        
        # サンプル画像で情報確認
        sample_paths = [
            self.source_dir / "normal" / "train",
            self.source_dir / "anomalous"
        ]
        
        for sample_dir in sample_paths:
            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.png"))
                if sample_files:
                    sample_info = self.get_image_info(sample_files[0])
                    if sample_info:
                        print(f"   📷 サンプル画像: {sample_files[0].name}")
                        print(f"      サイズ: {sample_info['size']} ({sample_info['size'][0]}×{sample_info['size'][1]})")
                        print(f"      モード: {sample_info['mode']}")
                        print(f"      形式: {sample_info['format']}")
                        break
    
    def process_mvtec_dataset(self):
        """MVTec ADデータセットの完全処理"""
        print(f"🚀 MVTec AD {self.dataset_name} データセット前処理開始")
        print(f"   ソース: {self.source_dir}")
        print(f"   出力: {self.output_dir}")
        print(f"   目標サイズ: {self.target_size[0]}×{self.target_size[1]}")
        
        # データセット分析
        self.analyze_dataset()
        
        start_time = datetime.now()
        
        # 1. Normal Training Data (293枚)
        train_normal_count = self.process_directory(
            "normal/train", "train/normal", "normal_train"
        )
        
        # 2. Normal Validation Data (33枚)
        val_normal_count = self.process_directory(
            "normal/vald", "validation/normal", "normal_val"
        )
        
        # 3. Anomalous Data (90枚) - 訓練用に分割
        anomalous_files = list((self.source_dir / "anomalous").glob("*.png"))
        
        # 異常データを7:3で分割（train:validation）
        np.random.seed(42)  # 再現性のため
        np.random.shuffle(anomalous_files)
        
        split_point = int(len(anomalous_files) * 0.7)
        train_anomalous = anomalous_files[:split_point]
        val_anomalous = anomalous_files[split_point:]
        
        # 異常データ - Training
        train_anomalous_count = 0
        print(f"\n📁 処理中: anomalous → train/anomalous")
        print(f"   ファイル数: {len(train_anomalous)}枚")
        
        for image_file in tqdm(train_anomalous, desc="処理中 anomalous_train"):
            output_file = self.output_dir / "train" / "anomalous" / image_file.name
            success, _ = self.resize_image(image_file, output_file)
            if success:
                train_anomalous_count += 1
                self.stats["processed_count"] += 1
        
        # 異常データ - Validation
        val_anomalous_count = 0
        print(f"\n📁 処理中: anomalous → validation/anomalous")
        print(f"   ファイル数: {len(val_anomalous)}枚")
        
        for image_file in tqdm(val_anomalous, desc="処理中 anomalous_val"):
            output_file = self.output_dir / "validation" / "anomalous" / image_file.name
            success, _ = self.resize_image(image_file, output_file)
            if success:
                val_anomalous_count += 1
                self.stats["processed_count"] += 1
        
        # 処理時間計算
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 統計情報更新
        self.stats["dataset_info"] = {
            "dataset_name": self.dataset_name,
            "original_total": train_normal_count + val_normal_count + len(anomalous_files),
            "processed_total": self.stats["processed_count"],
            "train_normal": train_normal_count,
            "train_anomalous": train_anomalous_count,
            "val_normal": val_normal_count,
            "val_anomalous": val_anomalous_count,
            "target_size": self.target_size,
            "processing_time": processing_time
        }
        
        # 結果報告
        self.print_summary()
        
        # 統計情報保存
        self.save_statistics()
        
        return self.stats
    
    def print_summary(self):
        """処理結果サマリー表示"""
        info = self.stats["dataset_info"]
        
        print("\n" + "="*60)
        print(f"🎉 MVTec AD {info['dataset_name']} 前処理完了")
        print("="*60)
        print(f"📊 処理結果:")
        print(f"   総処理数: {info['processed_total']}枚")
        print(f"   訓練用正常: {info['train_normal']}枚")
        print(f"   訓練用異常: {info['train_anomalous']}枚")
        print(f"   検証用正常: {info['val_normal']}枚")
        print(f"   検証用異常: {info['val_anomalous']}枚")
        print(f"   目標サイズ: {info['target_size'][0]}×{info['target_size'][1]}")
        print(f"   処理時間: {info['processing_time']:.1f}秒")
        
        if self.stats["processing_errors"]:
            print(f"\n⚠️  エラー: {len(self.stats['processing_errors'])}件")
        
        # 元サイズ統計
        if self.stats["original_sizes"]:
            unique_sizes = list(set(self.stats["original_sizes"]))
            print(f"📏 元画像サイズ: {unique_sizes}")
        
        print(f"\n📁 出力ディレクトリ: {self.output_dir}")
        print("="*60)
    
    def save_statistics(self):
        """統計情報をJSONファイルに保存"""
        stats_file = self.output_dir / "preprocessing_stats.json"
        
        # serializableに変換
        serializable_stats = {
            "dataset_info": self.stats["dataset_info"],
            "processed_count": self.stats["processed_count"],
            "original_sizes": [list(size) for size in set(self.stats["original_sizes"])],
            "processing_errors": self.stats["processing_errors"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📄 統計情報保存: {stats_file}")

def main():
    """メイン実行"""
    print("🚀 MVTec AD データセット前処理ツール")
    print("データセット: wallplugs")
    print("目標サイズ: 1024×1024")
    
    # 前処理実行
    preprocessor = MVTecPreprocessor(
        dataset_name="wallplugs",
        target_size=(1024, 1024)
    )
    
    stats = preprocessor.process_mvtec_dataset()
    
    # 追加情報
    print("\n💡 次のステップ:")
    print("1. data/processed/wallplugs/ で前処理済み画像を確認")
    print("2. MAD-FH システムでの学習・分析に使用")
    print("3. 他のデータセット（sheet_metal, wallnuts, fruit_jelly）も同様に処理")

if __name__ == "__main__":
    main()