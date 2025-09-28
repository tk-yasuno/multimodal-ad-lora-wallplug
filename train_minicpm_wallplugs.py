"""
MVTec AD Wallplugs × MiniCPM 異常検知モデル学習スクリプト
MiniCPM統合ハイブリッド異常検知モデルの学習実行
"""

import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.models.minicpm_autoencoder import MiniCPMAnomalyDetector, MiniCPMHybridAutoencoder
from src.utils.logger import setup_logger

class WallplugsDataset(Dataset):
    """MVTec AD Wallplugs データセット"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 画像パスとラベルを取得
        self.samples = []
        
        # 正常画像
        normal_dir = self.data_dir / split / "normal"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.png"):
                self.samples.append((str(img_path), 0))  # 0: normal
        
        # 異常画像
        anomalous_dir = self.data_dir / split / "anomalous"
        if anomalous_dir.exists():
            for img_path in anomalous_dir.glob("*.png"):
                self.samples.append((str(img_path), 1))  # 1: anomalous
        
        print(f"{split} dataset: {len(self.samples)} samples")
        print(f"  Normal: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Anomalous: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 画像読み込み
        image = Image.open(img_path).convert('RGB')
        
        # 前処理
        if self.transform:
            image = self.transform(image)
        else:
            # デフォルト変換
            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            'images': image,
            'anomaly_labels': torch.tensor(label, dtype=torch.float32),
            'image_path': img_path
        }

class WallplugsTrainer:
    """Wallplugsデータセット学習クラス"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ログ設定
        self.logger = setup_logger('wallplugs_trainer')
        
        # データ準備
        self.setup_data()
        
        # モデル初期化
        self.setup_model()
        
        # オプティマイザー設定
        self.setup_optimizer()
        
        # 記録用
        self.train_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'anomaly_loss': [],
            'validation_score': []
        }
    
    def _safe_item(self, value):
        """安全な値取得（tensorでもnumberでも対応）"""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:  # スカラーtensor
                return value.item()
            else:
                return float(value.mean().item())  # 複数要素の場合は平均
        else:
            return float(value)  # 既に数値の場合
    
    def setup_data(self):
        """データローダー準備"""
        data_dir = Path("data/processed/wallplugs")
        
        if not data_dir.exists():
            raise ValueError("前処理済みデータが見つかりません。preprocess_mvtec.pyを先に実行してください。")
        
        # データセット作成
        self.train_dataset = WallplugsDataset(data_dir, split='train')
        self.val_dataset = WallplugsDataset(data_dir, split='validation')
        
        # データローダー作成
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.logger.info(f"Data setup complete:")
        self.logger.info(f"  Train samples: {len(self.train_dataset)}")
        self.logger.info(f"  Val samples: {len(self.val_dataset)}")
        self.logger.info(f"  Batch size: {self.config['training']['batch_size']}")
    
    def setup_model(self):
        """モデル初期化"""
        model_config = {
            'input_channels': 3,
            'latent_dim': self.config['model']['latent_dim'],
            'input_size': (1024, 1024),
            'use_minicpm': self.config['model'].get('use_minicpm', True),
            'minicpm_weight': self.config['model'].get('minicpm_weight', 0.3),
            'anomaly_threshold': self.config['model']['anomaly_threshold']
        }
        
        self.detector = MiniCPMAnomalyDetector(model_config, device=self.device)
        
        # モデル情報表示
        total_params = sum(p.numel() for p in self.detector.model.parameters())
        trainable_params = sum(p.numel() for p in self.detector.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model setup complete:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  MiniCPM enabled: {model_config['use_minicpm']}")
    
    def setup_optimizer(self):
        """オプティマイザー設定"""
        self.optimizer = optim.AdamW(
            self.detector.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        self.logger.info(f"Optimizer setup complete:")
        self.logger.info(f"  LR: {self.config['training']['learning_rate']}")
        self.logger.info(f"  Weight decay: {self.config['training']['weight_decay']}")
    
    def train_epoch(self, epoch):
        """1エポック学習"""
        self.detector.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_anomaly_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # 学習ステップ
            losses, output = self.detector.train_step(batch)
            
            # バックプロパゲーション
            losses['total_loss'].backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.detector.model.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 損失蓄積（安全な値取得）
            total_loss += self._safe_item(losses['total_loss'])
            total_recon_loss += self._safe_item(losses['reconstruction_loss'])
            total_anomaly_loss += self._safe_item(losses['anomaly_loss'])
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                'Total': f"{self._safe_item(losses['total_loss']):.4f}",
                'Recon': f"{self._safe_item(losses['reconstruction_loss']):.4f}",
                'Anomaly': f"{self._safe_item(losses['anomaly_loss']):.4f}"
            })
        
        # エポック平均損失
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_anomaly_loss = total_anomaly_loss / num_batches
        
        # 記録
        self.train_history['total_loss'].append(avg_total_loss)
        self.train_history['reconstruction_loss'].append(avg_recon_loss)
        self.train_history['anomaly_loss'].append(avg_anomaly_loss)
        
        # スケジューラー更新
        self.scheduler.step()
        
        return avg_total_loss, avg_recon_loss, avg_anomaly_loss
    
    def validate(self):
        """検証"""
        self.detector.model.eval()
        
        val_loss = 0.0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                labels = batch['anomaly_labels']
                
                # 予測
                for i in range(images.shape[0]):
                    result = self.detector.predict(images[i:i+1])
                    predictions.append(result['anomaly_score'])
                    # tensorの場合のみitem()を使用
                    if isinstance(labels[i], torch.Tensor):
                        ground_truth.append(labels[i].item())
                    else:
                        ground_truth.append(labels[i])
                
                # 損失計算
                output = self.detector.model(images)
                losses = self.detector.model.compute_loss(
                    images, output, labels.to(self.device)
                )
                val_loss += self._safe_item(losses['total_loss'])
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        # 異常検知性能評価
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # ROC-AUC計算
        try:
            from sklearn.metrics import roc_auc_score, classification_report
            auc_score = roc_auc_score(ground_truth, predictions)
            
            # 分類レポート（閾値ベース）
            binary_predictions = predictions > self.detector.anomaly_threshold
            report = classification_report(
                ground_truth, 
                binary_predictions, 
                target_names=['Normal', 'Anomalous'],
                output_dict=True
            )
            
        except Exception as e:
            self.logger.warning(f"Metrics calculation failed: {e}")
            auc_score = 0.0
            report = {}
        
        validation_score = auc_score
        self.train_history['validation_score'].append(validation_score)
        
        return avg_val_loss, validation_score, report
    
    def save_checkpoint(self, epoch, is_best=False):
        """チェックポイント保存"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # モデル保存
        model_path = models_dir / f"minicpm_autoencoder_wallplugs_epoch{epoch+1}.pth"
        self.detector.save_model(model_path)
        
        if is_best:
            best_path = models_dir / "minicpm_autoencoder_wallplugs_best.pth"
            self.detector.save_model(best_path)
            self.logger.info(f"Best model saved: {best_path}")
        
        # 学習履歴保存
        history_path = models_dir / "training_history_wallplugs.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        return model_path
    
    def plot_training_history(self):
        """学習履歴可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失グラフ
        epochs = range(1, len(self.train_history['total_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.train_history['total_loss'], 'b-', label='Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.train_history['reconstruction_loss'], 'g-', label='Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, self.train_history['anomaly_loss'], 'r-', label='Anomaly Loss')
        axes[1, 0].set_title('Anomaly Detection Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        if self.train_history['validation_score']:
            axes[1, 1].plot(epochs, self.train_history['validation_score'], 'm-', label='Validation AUC')
            axes[1, 1].set_title('Validation AUC Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存
        plot_path = Path("models") / "training_history_wallplugs.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved: {plot_path}")
    
    def train(self):
        """学習実行"""
        self.logger.info("🚀 MiniCPM Anomaly Detection Training Started")
        self.logger.info("="*60)
        
        best_val_score = 0.0
        patience_counter = 0
        patience = self.config['training'].get('patience', 10)
        
        start_time = datetime.now()
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            self.logger.info("-" * 30)
            
            # 学習
            train_total_loss, train_recon_loss, train_anomaly_loss = self.train_epoch(epoch)
            
            # 検証
            val_loss, val_score, val_report = self.validate()
            
            # ログ出力
            self.logger.info(f"Train - Total: {train_total_loss:.4f}, Recon: {train_recon_loss:.4f}, Anomaly: {train_anomaly_loss:.4f}")
            self.logger.info(f"Val - Loss: {val_loss:.4f}, AUC: {val_score:.4f}")
            
            # ベストモデル判定
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                patience_counter = 0
                self.logger.info(f"🎉 New best model! AUC: {val_score:.4f}")
            else:
                patience_counter += 1
            
            # チェックポイント保存
            self.save_checkpoint(epoch, is_best)
            
            # 早期停止判定
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # 学習完了
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 Training Completed!")
        self.logger.info(f"Training time: {training_time:.1f} seconds")
        self.logger.info(f"Best validation AUC: {best_val_score:.4f}")
        
        # 学習履歴可視化
        self.plot_training_history()
        
        return best_val_score

def load_config():
    """設定読み込み"""
    default_config = {
        'model': {
            'latent_dim': 512,
            'use_minicpm': True,
            'minicpm_weight': 0.3,
            'anomaly_threshold': 0.1
        },
        'training': {
            'batch_size': 4,  # MiniCPMのため小さめ
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 10
        }
    }
    
    # 設定ファイルがあれば読み込み
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        
        # マージ
        if 'minicpm_training' in file_config:
            default_config.update(file_config['minicpm_training'])
    
    return default_config

def main():
    """メイン実行"""
    print("MVTec AD Wallplugs x MiniCPM 異常検知学習")
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
        trainer = WallplugsTrainer(config)
        best_score = trainer.train()
        
        print(f"\nTraining completed successfully!")
        print(f"   Best AUC Score: {best_score:.4f}")
        print(f"   Model saved: models/minicpm_autoencoder_wallplugs_best.pth")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()