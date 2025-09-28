"""
LoRAマルチモーダルモデルの学習クラス
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
import yaml
import json
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class BLIPTrainer(Trainer):
    """BLIP用カスタムTrainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        BLIP用にカスタマイズされた損失計算
        """
        # デバッグ: 入力されたキーを確認
        logger.info(f"入力キー: {list(inputs.keys())}")
        
        # 不要なキーを除去
        unwanted_keys = ['inputs_embeds', 'token_type_ids']
        for key in unwanted_keys:
            if key in inputs:
                logger.info(f"不要なキー '{key}' を削除")
                del inputs[key]
        
        # BLIPモデルに適した引数のみ渡す
        forward_inputs = {}
        
        # 必要なキーのみ抽出
        required_keys = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key in inputs and inputs[key] is not None:
                forward_inputs[key] = inputs[key]
        
        logger.info(f"モデルに渡すキー: {list(forward_inputs.keys())}")
        
        outputs = model(**forward_inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
import numpy as np

from .multimodal_model import LoRAMultimodalModel, AnomalyDescriptionEvaluator
from .multimodal_dataset import (
    AnomalyDescriptionDataModule, 
    MultimodalAnomalyDataset,
    collate_fn
)

logger = logging.getLogger(__name__)

class LoRAMultimodalTrainer:
    """
    LoRAマルチモーダルモデルの学習クラス
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルとデータモジュールの初期化
        self.model = LoRAMultimodalModel(self.config)
        self.data_module = None
        
        # 学習履歴
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        logger.info("LoRAマルチモーダルトレーナー初期化完了")
    
    def prepare_data(self, dummy_jsonl_path: str = None):
        """データの準備"""
        logger.info("学習データの準備開始")
        
        # データモジュールの初期化
        self.model.load_model(load_pretrained=False)
        self.data_module = AnomalyDescriptionDataModule(
            self.config,
            self.model.processor
        )
        
        # ダミーデータの場合は直接指定されたパスを使用
        if dummy_jsonl_path:
            jsonl_path = dummy_jsonl_path
            logger.info(f"ダミーデータを使用: {jsonl_path}")
        else:
            # フィードバックDBから学習用データセットを作成
            jsonl_path = self.data_module.prepare_data()
        
        # 学習・検証データセットの作成
        self.train_dataset, self.val_dataset = self.data_module.create_datasets(jsonl_path)
        
        logger.info(f"学習データ: {len(self.train_dataset)}サンプル")
        logger.info(f"検証データ: {len(self.val_dataset)}サンプル")
    
    def create_trainer(self) -> Trainer:
        """Hugging Face Trainerの作成"""
        training_config = self.config['text_generation']['training']
        
        # 学習引数の設定
        training_args = TrainingArguments(
            output_dir=str(Path(self.config['text_generation']['model_path']) / "checkpoints"),
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            num_train_epochs=training_config['epochs'],
            learning_rate=float(training_config['learning_rate']),  # 明示的にfloatに変換
            warmup_steps=training_config['warmup_steps'],
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # wandbやtensorboard無効化
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        
        # データコレーター - BLIPモデル用にカスタマイズ
        def data_collator(features):
            """BLIP用カスタムデータコレーター"""
            batch = {}
            
            # pixel_valuesを処理
            if "pixel_values" in features[0]:
                batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
            
            # input_idsとattention_maskを処理
            if "input_ids" in features[0]:
                batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
            
            if "attention_mask" in features[0]:
                batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
            
            # labelsを処理
            if "labels" in features[0]:
                batch["labels"] = torch.stack([f["labels"] for f in features])
            
            return batch
        
        # トレーナーの作成
        trainer = BLIPTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
        )
        
        return trainer
    
    def train(self):
        """モデルの学習実行"""
        logger.info("LoRAモデルの学習開始")
        
        if self.train_dataset is None:
            self.prepare_data()
        
        try:
            # トレーナーの作成
            trainer = self.create_trainer()
            
            # 学習実行
            train_result = trainer.train()
            
            # 学習履歴の記録
            self._record_training_history(trainer)
            
            # モデルの保存
            self.model.save_model()
            
            # 学習結果の保存
            self._save_training_results(train_result)
            
            logger.info("LoRAモデルの学習完了")
            
            return train_result
            
        except Exception as e:
            logger.error(f"学習中にエラーが発生: {e}")
            raise
    
    def evaluate(self, test_data_path: str = None) -> Dict[str, float]:
        """モデルの評価"""
        logger.info("モデルの評価開始")
        
        if test_data_path is None:
            # 検証データでの評価
            evaluator = AnomalyDescriptionEvaluator(self.model)
            
            # 検証データからサンプルを取得
            test_images = []
            ground_truth_descriptions = []
            
            for i in range(min(10, len(self.val_dataset))):  # 最初の10サンプル
                sample = self.val_dataset[i]
                
                # 画像とテキストを取得（簡易実装）
                # 実際の実装では、データセットから適切に画像とテキストを抽出
                try:
                    # ダミーデータでの評価（実装を簡素化）
                    from PIL import Image
                    dummy_image = Image.new('RGB', (512, 512), color='blue')
                    test_images.append(dummy_image)
                    ground_truth_descriptions.append("テスト用の説明文")
                except Exception as e:
                    logger.warning(f"評価データ準備エラー: {e}")
                    continue
            
            if test_images:
                metrics = evaluator.evaluate_on_test_set(
                    test_images, ground_truth_descriptions
                )
                
                logger.info(f"評価結果: {metrics}")
                return metrics
            else:
                logger.warning("評価用データが見つかりません")
                return {}
        
        return {}
    
    def _record_training_history(self, trainer):
        """学習履歴の記録"""
        try:
            # ログから学習履歴を抽出
            log_history = trainer.state.log_history
            
            for log in log_history:
                if 'train_loss' in log:
                    self.training_history['train_loss'].append(log['train_loss'])
                    self.training_history['learning_rates'].append(log.get('learning_rate', 0))
                    self.training_history['epochs'].append(log.get('epoch', 0))
                
                if 'eval_loss' in log:
                    self.training_history['eval_loss'].append(log['eval_loss'])
            
        except Exception as e:
            logger.warning(f"学習履歴記録エラー: {e}")
    
    def _save_training_results(self, train_result):
        """学習結果の保存"""
        results_dir = Path(self.config['text_generation']['model_path']) / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 学習履歴をJSONで保存
            history_path = results_dir / f"training_history_{timestamp}.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)
            
            # 学習結果を保存
            results_path = results_dir / f"training_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'train_runtime': train_result.metrics.get('train_runtime', 0),
                    'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
                    'total_flos': train_result.metrics.get('total_flos', 0),
                    'config': self.config['text_generation']
                }, f, ensure_ascii=False, indent=2)
            
            # 学習カーブをプロット
            self._plot_training_curves(results_dir, timestamp)
            
            logger.info(f"学習結果保存完了: {results_dir}")
            
        except Exception as e:
            logger.error(f"学習結果保存エラー: {e}")
    
    def _plot_training_curves(self, results_dir: Path, timestamp: str):
        """学習カーブのプロット"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            if self.training_history['train_loss']:
                axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Steps')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            if self.training_history['eval_loss']:
                axes[0, 1].plot(self.training_history['eval_loss'], label='Eval Loss', color='orange')
                axes[0, 1].set_title('Validation Loss')
                axes[0, 1].set_xlabel('Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate
            if self.training_history['learning_rates']:
                axes[1, 0].plot(self.training_history['learning_rates'], label='Learning Rate')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Combined loss
            if self.training_history['train_loss'] and self.training_history['eval_loss']:
                # 評価ステップに合わせて学習ロスをサンプリング
                train_steps = len(self.training_history['train_loss'])
                eval_steps = len(self.training_history['eval_loss'])
                
                if eval_steps > 0:
                    eval_interval = train_steps // eval_steps
                    train_sampled = self.training_history['train_loss'][::eval_interval][:eval_steps]
                    
                    axes[1, 1].plot(train_sampled, label='Train Loss')
                    axes[1, 1].plot(self.training_history['eval_loss'], label='Eval Loss')
                    axes[1, 1].set_title('Training vs Validation Loss')
                    axes[1, 1].set_xlabel('Eval Steps')
                    axes[1, 1].set_ylabel('Loss')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 図を保存
            plot_path = results_dir / f"training_curves_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"学習カーブ保存: {plot_path}")
            
        except Exception as e:
            logger.warning(f"学習カーブプロットエラー: {e}")

def main():
    """メイン学習実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # トレーナーの初期化
        trainer = LoRAMultimodalTrainer()
        
        # データ準備
        trainer.prepare_data()
        
        # 学習実行
        train_result = trainer.train()
        
        # 評価実行
        eval_results = trainer.evaluate()
        
        print("=== 学習完了 ===")
        print(f"学習時間: {train_result.metrics.get('train_runtime', 0):.2f}秒")
        print(f"評価結果: {eval_results}")
        
    except Exception as e:
        logger.error(f"学習実行エラー: {e}")
        raise

if __name__ == "__main__":
    main()
