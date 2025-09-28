"""
MAD-FH: Image Preprocessing Module
画像のリサイズ、正規化、Augmentation、サンプリングを実行
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
from tqdm import tqdm
import json

from .metadata_manager import ImageMetadataDB


class ImagePreprocessor:
    """画像前処理クラス"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # デフォルト設定
            self.config = {
                'data': {
                    'preprocessing': {
                        'target_size': [512, 512],
                        'normalize_mean': [0.485, 0.456, 0.406],
                        'normalize_std': [0.229, 0.224, 0.225]
                    }
                }
            }
        
        self.target_size = tuple(self.config['data']['preprocessing']['target_size'])
        self.normalize_mean = self.config['data']['preprocessing']['normalize_mean']
        self.normalize_std = self.config['data']['preprocessing']['normalize_std']
        
        # Albumentations変換パイプラインの定義
        self._setup_transforms()
    
    def _setup_transforms(self):
        """変換パイプラインのセットアップ"""
        
        # 基本的な前処理（リサイズ + 正規化）
        self.basic_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2()
        ])
        
        # 軽微なAugmentation付き前処理
        self.augment_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            
            # 軽微なAugmentation
            A.OneOf([
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),
            ], p=0.5),
            
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2()
        ])
        
        # 評価用（Augmentationなし）
        self.eval_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str, apply_augmentation: bool = False) -> torch.Tensor:
        """
        単一画像の前処理
        
        Args:
            image_path: 画像ファイルのパス
            apply_augmentation: Augmentationを適用するか
            
        Returns:
            前処理済みテンソル
        """
        # 画像読み込み（OpenCV -> RGB変換）
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 変換適用
        if apply_augmentation:
            transformed = self.augment_transform(image=image)
        else:
            transformed = self.basic_transform(image=image)
        
        return transformed['image']
    
    def preprocess_batch(self, 
                        image_paths: List[str], 
                        output_dir: str,
                        apply_augmentation: bool = False,
                        save_tensors: bool = True) -> List[str]:
        """
        画像のバッチ前処理
        
        Args:
            image_paths: 画像パスのリスト
            output_dir: 出力ディレクトリ
            apply_augmentation: Augmentationを適用するか
            save_tensors: テンソルを保存するか
            
        Returns:
            処理済み画像のパスリスト
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_paths = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # 前処理実行
                tensor = self.preprocess_image(image_path, apply_augmentation)
                
                # ファイル名生成
                original_name = Path(image_path).stem
                if apply_augmentation:
                    output_name = f"{original_name}_aug.pt"
                else:
                    output_name = f"{original_name}.pt"
                
                output_path = output_dir / output_name
                
                if save_tensors:
                    # テンソル保存
                    torch.save(tensor, output_path)
                    processed_paths.append(str(output_path))
                else:
                    processed_paths.append(tensor)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return processed_paths
    
    def create_sample_dataset(self, 
                             image_list: List[str], 
                             sample_size: int, 
                             random_seed: int = 42) -> List[str]:
        """
        サンプルデータセットの作成
        
        Args:
            image_list: 全画像パスのリスト
            sample_size: サンプル数
            random_seed: ランダムシード
            
        Returns:
            サンプリングされた画像パスのリスト
        """
        if len(image_list) <= sample_size:
            return image_list
        
        random.seed(random_seed)
        return random.sample(image_list, sample_size)


class NormalImageDataset(Dataset):
    """正常画像データセット"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 preprocessor: ImagePreprocessor,
                 apply_augmentation: bool = False):
        """
        Args:
            image_paths: 画像パスのリスト
            preprocessor: 前処理器
            apply_augmentation: Augmentationを適用するか
        """
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        self.apply_augmentation = apply_augmentation
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            tensor = self.preprocessor.preprocess_image(
                image_path, 
                apply_augmentation=self.apply_augmentation
            )
            return tensor, image_path
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # エラー時はゼロテンソルを返す
            return torch.zeros(3, *self.preprocessor.target_size), image_path


def create_preprocessing_pipeline(config_path: str, 
                                db_path: str,
                                output_dir: str,
                                sample_size: int = None) -> Dict:
    """
    前処理パイプラインの実行
    
    Args:
        config_path: 設定ファイルのパス
        db_path: メタデータDBのパス
        output_dir: 出力ディレクトリ
        sample_size: サンプル数（指定しない場合は全データ）
        
    Returns:
        処理結果の統計情報
    """
    
    # 設定読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # メタデータDB接続
    db = ImageMetadataDB(db_path)
    
    # 未処理画像の取得
    unprocessed_images = db.list_images(is_processed=False)
    
    if not unprocessed_images:
        print("No unprocessed images found.")
        return {"processed_count": 0}
    
    print(f"Found {len(unprocessed_images)} unprocessed images")
    
    # 前処理器の初期化
    preprocessor = ImagePreprocessor(config_path)
    
    # 画像パスのリスト取得
    image_paths = [img['image_path'] for img in unprocessed_images]
    
    # サンプリング（指定された場合）
    if sample_size and sample_size < len(image_paths):
        random_seed = config.get('data', {}).get('sampling', {}).get('random_seed', 42)
        image_paths = preprocessor.create_sample_dataset(
            image_paths, sample_size, random_seed
        )
        print(f"Sampled {len(image_paths)} images for processing")
    
    # バッチ前処理実行
    processed_paths = preprocessor.preprocess_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        apply_augmentation=False,  # 初期処理ではAugmentationなし
        save_tensors=True
    )
    
    # データベースの処理済みフラグ更新
    for img in unprocessed_images:
        if img['image_path'] in image_paths:
            db.mark_processed(img['id'], True)
    
    # 処理結果の保存
    result_info = {
        "total_images": len(unprocessed_images),
        "processed_count": len(processed_paths),
        "sample_size": sample_size,
        "output_directory": str(output_dir),
        "target_size": preprocessor.target_size,
        "processed_paths": processed_paths[:10]  # 最初の10個のパスを保存
    }
    
    # 結果をJSONで保存
    output_dir = Path(output_dir)
    with open(output_dir / "preprocessing_info.json", 'w', encoding='utf-8') as f:
        json.dump(result_info, f, indent=2, ensure_ascii=False)
    
    print(f"Preprocessing completed: {len(processed_paths)} images processed")
    
    return result_info


if __name__ == "__main__":
    # テスト実行
    import tempfile
    
    # 設定ファイルのパス（テスト用）
    config_path = "config/config.yaml"
    
    # 一時的なテスト
    preprocessor = ImagePreprocessor()
    
    print("Image Preprocessor initialized")
    print(f"Target size: {preprocessor.target_size}")
    print(f"Normalization - Mean: {preprocessor.normalize_mean}, Std: {preprocessor.normalize_std}")
    
    # サンプルデータセット作成のテスト
    sample_paths = [f"image_{i}.jpg" for i in range(100)]
    sampled = preprocessor.create_sample_dataset(sample_paths, 20)
    print(f"Original: {len(sample_paths)}, Sampled: {len(sampled)}")
