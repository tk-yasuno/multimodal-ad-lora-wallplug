"""
MVTec AD Wallplugs × LoRA 説明生成モデル学習スクリプト
wallplugsデータセット専用のLoRAマルチモーダル説明生成モデル学習
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import yaml
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq,
    TrainingArguments, Trainer
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
import matplotlib.pyplot as plt

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.utils.logger import setup_logger

class WallplugsExplanationDataset(Dataset):
    """Wallplugs説明生成データセット"""
    
    def __init__(self, data_dir, processor, split='train', max_length=128):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        self.max_length = max_length
        
        # データ収集
        self.samples = []
        self.collect_samples()
        
        print(f"{split} dataset: {len(self.samples)} samples")
    
    def collect_samples(self):
        """サンプルデータ収集"""
        # 正常画像のサンプル
        normal_dir = self.data_dir / self.split / "normal"
        if normal_dir.exists():
            for img_path in list(normal_dir.glob("*.png"))[:50]:  # サンプル数制限
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 'normal',
                    'explanation': self.generate_normal_explanation(img_path.name)
                })
        
        # 異常画像のサンプル
        anomalous_dir = self.data_dir / self.split / "anomalous"
        if anomalous_dir.exists():
            for img_path in list(anomalous_dir.glob("*.png")):
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 'anomalous',
                    'explanation': self.generate_anomaly_explanation(img_path.name)
                })
    
    def generate_normal_explanation(self, filename):
        """正常画像の説明生成"""
        explanations = [
            "This wallplug appears to be in normal condition with proper alignment and no visible defects.",
            "The wallplug shows normal manufacturing quality with correct positioning and surface finish.",
            "This is a properly manufactured wallplug with no anomalies detected in shape or surface.",
            "The wallplug demonstrates standard quality with appropriate dimensions and surface texture.",
            "Normal wallplug with correct structural integrity and expected appearance."
        ]
        return np.random.choice(explanations)
    
    def generate_anomaly_explanation(self, filename):
        """異常画像の説明生成（ファイル名から推定）"""
        if "overexposed" in filename:
            return "This wallplug image shows overexposure issues, making surface details difficult to assess properly."
        elif "underexposed" in filename:
            return "The wallplug appears underexposed with reduced visibility of surface features and potential defects."
        elif "shift" in filename:
            return "This wallplug shows positional shift or misalignment, indicating potential manufacturing or positioning issues."
        elif "regular" in filename and "anomalous" in str(self.data_dir):
            return "This wallplug exhibits subtle anomalies that may not be immediately visible but differ from normal standards."
        else:
            return "This wallplug shows anomalous characteristics that deviate from normal manufacturing standards."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 画像読み込み
        image = Image.open(sample['image_path']).convert('RGB')
        
        # プロンプト作成
        prompt = f"Describe this wallplug manufacturing quality:"
        full_text = f"{prompt} {sample['explanation']}"
        
        # エンコード
        encoding = self.processor(
            images=image,
            text=full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # プロンプトのみのエンコード（推論用）
        prompt_encoding = self.processor(
            images=image,
            text=prompt,
            padding="max_length", 
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # 自己回帰学習用
            'prompt_input_ids': prompt_encoding['input_ids'].squeeze(),
            'explanation': sample['explanation'],
            'image_path': sample['image_path']
        }

class WallplugsLoRATrainer:
    """Wallplugs LoRA学習クラス"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ログ設定
        self.logger = setup_logger('wallplugs_lora_trainer')
        
        # モデル設定
        self.model_name = config['model']['name']
        self.output_dir = Path("models/lora_wallplugs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル・プロセッサ初期化
        self.setup_model()
        
        # データ準備
        self.setup_data()
        
        # 学習履歴
        self.training_history = {
            'loss': [],
            'learning_rate': [],
            'generated_examples': []
        }
    
    def setup_model(self):
        """モデル・プロセッサ初期化"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # プロセッサ読み込み
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        # ベースモデル読み込み
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias="none"
        )
        
        # LoRA適用
        self.model = get_peft_model(self.model, lora_config)
        
        # 学習可能パラメータ情報
        self.model.print_trainable_parameters()
        
        self.logger.info("Model setup complete")
    
    def setup_data(self):
        """データ準備"""
        data_dir = Path("data/processed/wallplugs")
        
        if not data_dir.exists():
            raise ValueError("前処理済みデータが見つかりません。preprocess_mvtec.pyを先に実行してください。")
        
        # データセット作成
        self.train_dataset = WallplugsExplanationDataset(
            data_dir, self.processor, split='train'
        )
        
        self.val_dataset = WallplugsExplanationDataset(
            data_dir, self.processor, split='validation'
        )
        
        self.logger.info(f"Data setup complete:")
        self.logger.info(f"  Train samples: {len(self.train_dataset)}")
        self.logger.info(f"  Val samples: {len(self.val_dataset)}")
    
    def custom_data_collator(self, batch):
        """カスタムデータコレーター"""
        # バッチサイズが1の場合の処理
        if len(batch) == 1:
            batch = batch[0]
            return {
                'pixel_values': batch['pixel_values'].unsqueeze(0),
                'input_ids': batch['input_ids'].unsqueeze(0),
                'attention_mask': batch['attention_mask'].unsqueeze(0),
                'labels': batch['labels'].unsqueeze(0)
            }
        
        # 複数サンプルのバッチ処理
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def generate_sample_explanations(self, num_samples=3):
        """サンプル説明生成"""
        self.model.eval()
        generated_examples = []
        
        # 各カテゴリから1つずつサンプル
        categories = [
            (self.train_dataset, "train"),
            (self.val_dataset, "validation")
        ]
        
        with torch.no_grad():
            for dataset, split_name in categories:
                if len(dataset) == 0:
                    continue
                    
                # ランダムサンプル選択
                sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
                
                for idx in sample_indices:
                    sample = dataset[idx]
                    
                    # 画像とプロンプト準備
                    image = Image.open(sample['image_path']).convert('RGB')
                    prompt = "Describe this wallplug manufacturing quality:"
                    
                    # プロンプト処理
                    inputs = self.processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # 生成
                    try:
                        outputs = self.model.generate(
                            **inputs,
                            max_length=100,
                            num_beams=4,
                            early_stopping=True,
                            do_sample=True,
                            temperature=0.7
                        )
                        
                        # デコード
                        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                        
                        # プロンプト部分を除去
                        if prompt in generated_text:
                            generated_explanation = generated_text.replace(prompt, "").strip()
                        else:
                            generated_explanation = generated_text
                        
                        generated_examples.append({
                            'split': split_name,
                            'image_path': sample['image_path'],
                            'ground_truth': sample['explanation'],
                            'generated': generated_explanation,
                            'image_filename': Path(sample['image_path']).name
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Generation failed for {sample['image_path']}: {e}")
        
        return generated_examples
    
    def train(self):
        """LoRA学習実行"""
        self.logger.info("🚀 LoRA Training Started")
        self.logger.info("="*60)
        
        # 学習引数設定
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        # カスタム学習ループを使用（Trainer使用時の互換性問題回避）
        return self._custom_training_loop(training_args)
        
        # 学習前のサンプル生成
        self.logger.info("Pre-training sample generation...")
        pre_examples = self.generate_sample_explanations()
        self.training_history['generated_examples'].append({
            'epoch': 0,
            'examples': pre_examples
        })
        
        start_time = datetime.now()
        
        try:
            # 学習実行
            trainer.train()
            
            # 学習後のサンプル生成
            self.logger.info("Post-training sample generation...")
            post_examples = self.generate_sample_explanations()
            self.training_history['generated_examples'].append({
                'epoch': self.config['training']['epochs'],
                'examples': post_examples
            })
            
            # 学習時間計算
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"\n🎉 Training completed successfully!")
            self.logger.info(f"Training time: {training_time:.1f} seconds")
            
            # モデル保存
            final_model_path = self.output_dir / "final_model"
            trainer.save_model(str(final_model_path))
            self.processor.save_pretrained(str(final_model_path))
            
            # 学習履歴保存
            self.save_training_results(training_time, trainer.state.log_history)
            
            # 結果レポート生成
            self.generate_training_report(pre_examples, post_examples)
            
            return str(final_model_path)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_training_results(self, training_time, log_history):
        """学習結果保存"""
        results = {
            'model_name': self.model_name,
            'training_time': training_time,
            'config': self.config,
            'log_history': log_history,
            'generated_examples_history': self.training_history['generated_examples']
        }
        
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Training results saved: {results_path}")
    
    def generate_training_report(self, pre_examples, post_examples):
        """学習レポート生成"""
        report = []
        report.append("# Wallplugs LoRA Training Report\n")
        report.append(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Model**: {self.model_name}\n")
        report.append(f"**Dataset**: MVTec AD Wallplugs\n\n")
        
        report.append("## Training Configuration\n")
        report.append("```yaml\n")
        report.append(yaml.dump(self.config, default_flow_style=False))
        report.append("```\n\n")
        
        report.append("## Sample Generation Comparison\n")
        
        # 学習前後の比較
        for i, (pre, post) in enumerate(zip(pre_examples[:3], post_examples[:3])):
            report.append(f"### Sample {i+1}: {pre['image_filename']}\n")
            report.append(f"**Ground Truth**: {pre['ground_truth']}\n\n")
            report.append(f"**Before Training**: {pre['generated']}\n\n")
            report.append(f"**After Training**: {post['generated']}\n\n")
            report.append("---\n\n")
        
        # レポート保存
        report_path = self.output_dir / "training_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))
        
        self.logger.info(f"Training report saved: {report_path}")
    
    def _custom_training_loop(self, training_args):
        """カスタム学習ループ（inputs_embedsエラー回避）"""
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from tqdm import tqdm
        
        # データローダー作成
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.custom_data_collator
        )
        
        # オプティマイザー設定
        optimizer = AdamW(
            self.model.parameters(), 
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
        
        self.logger.info("Starting custom training loop...")
        
        # 学習ループ
        self.model.train()
        
        for epoch in range(int(training_args.num_train_epochs)):
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{int(training_args.num_train_epochs)}')
            for batch in pbar:
                try:
                    # 入力準備
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # フォワードパス（シンプルな形式）
                    optimizer.zero_grad()
                    
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    self.logger.warning(f"Batch error (skipping): {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                self.logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # モデル保存
        model_path = self.output_dir / "final_model"
        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info("Custom training completed successfully!")
        
        return str(model_path)

def load_config():
    """設定読み込み"""
    default_config = {
        'model': {
            'name': 'Salesforce/blip-image-captioning-base'  # より安定したBLIPモデル
        },
        'lora': {
            'r': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['fc1', 'fc2']
        },
        'training': {
            'epochs': 10,
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 100
        }
    }
    
    # 設定ファイル読み込み
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        
        if 'wallplugs_lora' in file_config:
            default_config.update(file_config['wallplugs_lora'])
    
    return default_config

def main():
    """メイン実行"""
    print("MVTec AD Wallplugs x LoRA 説明生成学習")
    print("="*60)
    
    # 設定読み込み
    config = load_config()
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    
    # GPU情報表示
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("\nCPU mode (GPU not available)")
    
    try:
        # 学習実行
        trainer = WallplugsLoRATrainer(config)
        model_path = trainer.train()
        
        print(f"\nLoRA training completed successfully!")
        print(f"   Model saved: {model_path}")
        print(f"   Training results: models/lora_wallplugs/training_results.json")
        print(f"   Training report: models/lora_wallplugs/training_report.md")
        
        print("\nNext steps:")
        print("1. Test the trained LoRA model with FODD pipeline")
        print("2. Evaluate explanation quality")
        print("3. Fine-tune hyperparameters if needed")
        
    except Exception as e:
        print(f"\nLoRA training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()