"""
マルチモーダルデータセット処理
画像+テキストペアの学習データセット作成
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MultimodalAnomalyDataset(Dataset):
    """
    画像と異常説明テキストのペアデータセット
    """
    
    def __init__(
        self,
        jsonl_path: str,
        processor,
        image_root: str = "",
        max_length: int = 256,
        split: str = "train"
    ):
        """
        Args:
            jsonl_path: フィードバックデータのJSONLファイルパス
            processor: 画像・テキスト処理用のプロセッサ
            image_root: 画像ファイルのルートディレクトリ
            max_length: テキストの最大長
            split: データセットの分割（train/val/test）
        """
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.split = split
        
        # JSONLファイルからデータを読み込み
        self.data = self._load_jsonl_data(jsonl_path)
        
        # 異常データのみをフィルタリング
        self.anomaly_data = [
            item for item in self.data 
            if item.get('is_anomaly', False) and item.get('anomaly_description', '').strip()
        ]
        
        logger.info(f"総データ数: {len(self.data)}")
        logger.info(f"異常データ数: {len(self.anomaly_data)}")
        logger.info(f"分割: {split}")
    
    def _load_jsonl_data(self, jsonl_path: str) -> List[Dict]:
        """JSONLファイルからデータを読み込み"""
        data = []
        
        if not os.path.exists(jsonl_path):
            logger.warning(f"JSONLファイルが見つかりません: {jsonl_path}")
            return data
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"行 {line_num} の解析エラー: {e}")
                        continue
        except Exception as e:
            logger.error(f"JSONLファイル読み込みエラー: {e}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.anomaly_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """データセットからアイテムを取得"""
        item = self.anomaly_data[idx]
        
        # 画像の読み込み
        image_path = self._get_image_path(item['image_path'])
        image = self._load_image(image_path)
        
        # テキストの準備
        anomaly_text = self._prepare_text(item)
        
        # プロセッサでエンコード
        try:
            encoding = self.processor(
                images=image,
                text=anomaly_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # テンソルの次元を調整
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].squeeze(0)
            
            # ラベル（生成対象）も追加
            encoding['labels'] = encoding['input_ids'].clone()
            
            return encoding
            
        except Exception as e:
            logger.error(f"エンコーディングエラー (idx={idx}): {e}")
            # エラーの場合はダミーデータを返す
            return self._get_dummy_data()
    
    def _get_image_path(self, image_path: str) -> Path:
        """画像パスの正規化"""
        image_path = Path(image_path)
        
        # 相対パスの場合は image_root と結合
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        
        return image_path
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """画像を読み込み"""
        try:
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                return image
            else:
                logger.warning(f"画像ファイルが見つかりません: {image_path}")
                # ダミー画像を生成
                return Image.new('RGB', (512, 512), color='gray')
        except Exception as e:
            logger.error(f"画像読み込みエラー: {e}")
            return Image.new('RGB', (512, 512), color='gray')
    
    def _prepare_text(self, item: Dict) -> str:
        """テキストの準備とフォーマット"""
        anomaly_type = item.get('anomaly_type', '不明')
        description = item.get('anomaly_description', '')
        confidence = item.get('confidence_level', 3)
        
        # プロンプト形式でテキストを構成
        text = f"この画像には{anomaly_type}の異常が見られます。"
        
        if description.strip():
            text += f" 詳細: {description.strip()}"
        
        text += f" 確信度: {confidence}/5"
        
        return text
    
    def _get_dummy_data(self) -> Dict[str, torch.Tensor]:
        """エラー時のダミーデータ"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'pixel_values': torch.zeros(3, 224, 224, dtype=torch.float),
            'labels': torch.zeros(self.max_length, dtype=torch.long)
        }

class AnomalyDescriptionDataModule:
    """
    異常説明生成用のデータモジュール
    """
    
    def __init__(
        self,
        config: Dict,
        processor,
        feedback_db_path: str = None
    ):
        self.config = config
        self.processor = processor
        self.feedback_db_path = feedback_db_path or config['text_generation']['data']['feedback_db_path']
        
    def prepare_data(self) -> str:
        """フィードバックDBから学習用データセットを作成"""
        from src.ui.feedback_manager import FeedbackDataManager
        
        # フィードバックマネージャーの初期化
        feedback_manager = FeedbackDataManager(self.feedback_db_path)
        
        # 学習用データセットの作成
        output_dir = Path(self.config['text_generation']['data']['training_dataset_path']).parent
        dataset_info = feedback_manager.create_training_dataset(str(output_dir))
        
        logger.info(f"学習用データセット作成完了: {dataset_info['total_samples']}サンプル")
        
        return dataset_info['jsonl_path']
    
    def create_datasets(self, jsonl_path: str) -> Tuple[Dataset, Dataset]:
        """学習・検証データセットの作成"""
        
        # 全データを読み込み
        full_dataset = MultimodalAnomalyDataset(
            jsonl_path=jsonl_path,
            processor=self.processor,
            max_length=self.config['text_generation']['training']['max_length']
        )
        
        # 学習・検証に分割
        val_split = self.config['text_generation']['data']['validation_split']
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"学習データ: {len(train_dataset)}, 検証データ: {len(val_dataset)}")
        
        return train_dataset, val_dataset

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    バッチ処理用のコレート関数
    """
    # バッチ内の全アイテムのキーを統一
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'pixel_values':
            # 画像データはそのままスタック
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            # テキストデータはパディングしてスタック
            tensors = [item[key] for item in batch]
            collated[key] = torch.stack(tensors)
    
    return collated

# デバッグ用の関数
def test_dataset():
    """データセットのテスト"""
    import yaml
    from transformers import BlipProcessor
    
    # 設定読み込み
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # プロセッサの初期化
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # データセットの作成
    dataset = MultimodalAnomalyDataset(
        jsonl_path="data/feedback/training_dataset/training_data.jsonl",
        processor=processor,
        image_root="data/images"
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    
    if len(dataset) > 0:
        # 最初のアイテムをテスト
        item = dataset[0]
        print("サンプルアイテム:")
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

if __name__ == "__main__":
    test_dataset()
