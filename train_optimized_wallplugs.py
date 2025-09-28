"""
MVTec AD Wallplugs 改良版高性能学習システム
全416枚データ対応 - 性能最適化版
"""

import sys
import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class WallplugsOptimizedDataset(Dataset):
    """最適化Wallplugsデータセット"""
    
    def __init__(self, data_dir, split='train', transform=None, image_size=512):
        """
        Args:
            data_dir: data/processed/wallplugs
            split: 'train' または 'validation'
            transform: 画像変換
            image_size: 画像サイズ（512で高速化）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # データパス構築
        split_dir = self.data_dir / split
        normal_dir = split_dir / "normal"
        anomalous_dir = split_dir / "anomalous"
        
        # 画像パス収集
        self.image_paths = []
        self.labels = []
        
        # 正常画像
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(0)  # 正常=0
        
        # 異常画像
        if anomalous_dir.exists():
            for img_path in anomalous_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(1)  # 異常=1
        
        print(f"[DATA] {split} dataset: {len(self.image_paths)} samples")
        if split == 'train':
            normal_count = len(list(normal_dir.glob("*.png"))) if normal_dir.exists() else 0
            anomalous_count = len(list(anomalous_dir.glob("*.png"))) if anomalous_dir.exists() else 0
            print(f"  - Normal: {normal_count} samples")
            print(f"  - Anomalous: {anomalous_count} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 画像読み込み（512x512で高速化）
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class OptimizedAnomalyEncoder(nn.Module):
    """最適化異常検知エンコーダ（512x512対応）"""
    
    def __init__(self, input_size=512, latent_dim=256):
        super().__init__()
        
        # エンコーダ（512x512 → 256次元潜在表現）
        self.encoder = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 256 -> 128
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # 潜在表現（4x4x512 → latent_dim）
        self.latent_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, latent_dim),
            nn.ReLU()
        )
        
        # 潜在表現復元（latent_dim → 4x4x512）
        self.latent_restore = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.ReLU()
        )
        
        # デコーダ（4x4x512 → 512x512x3）
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 8 -> 16
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 16 -> 32
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 64 -> 128
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 128 -> 256
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 256 -> 512
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # エンコード
        encoded = self.encoder(x)
        
        # 潜在表現
        latent = self.latent_fc(encoded)
        
        # 潜在表現復元
        restored = self.latent_restore(latent)
        restored = restored.view(-1, 512, 4, 4)
        
        # デコード
        decoded = self.decoder(restored)
        
        return decoded, latent

class OptimizedWallplugsTrainer:
    """最適化416枚学習クラス"""
    
    def __init__(self, data_dir="data/processed/wallplugs", 
                 batch_size=16, learning_rate=0.0002, device=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # データ変換（512x512で高速化）
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # データセット準備
        self.train_dataset = WallplugsOptimizedDataset(
            self.data_dir, split='train', transform=self.transform
        )
        self.val_dataset = WallplugsOptimizedDataset(
            self.data_dir, split='validation', transform=self.transform
        )
        
        # データローダー（並列処理で高速化）
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=2, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # 最適化モデル準備
        self.model = OptimizedAnomalyEncoder(latent_dim=256).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                                   betas=(0.5, 0.999), weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # 損失関数（異常検知用）
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = nn.L1Loss()
        
        # 学習履歴
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_aucs': [],
            'best_auc': 0.0,
            'best_epoch': 0
        }
        
        # モデル情報表示
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Device: {self.device}")
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        print(f"Batch size: {batch_size}")
        print(f"Image size: 512x512 (optimized)")
        print(f"Total parameters: {total_params:,}")
    
    def compute_anomaly_loss(self, original, reconstructed):
        """異常検知用損失計算"""
        # 再構成損失
        recon_loss = self.reconstruction_loss(reconstructed, original)
        
        # 知覚損失（L1）
        percept_loss = self.perceptual_loss(reconstructed, original)
        
        # 総合損失
        total_loss = recon_loss + 0.1 * percept_loss
        
        return total_loss, recon_loss, percept_loss
    
    def evaluate_anomaly_detection(self, data_loader):
        """異常検知評価（最適化版）"""
        self.model.eval()
        all_reconstruction_errors = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.cpu().numpy()
                
                # 前向き計算
                reconstructed, latent = self.model(images)
                
                # バッチごとの再構成誤差
                batch_loss, recon_loss, percept_loss = self.compute_anomaly_loss(images, reconstructed)
                total_loss += batch_loss.item()
                
                # サンプルごとの再構成誤差計算
                batch_size = images.size(0)
                for i in range(batch_size):
                    img = images[i:i+1]
                    recon = reconstructed[i:i+1]
                    error = self.reconstruction_loss(img, recon)
                    all_reconstruction_errors.append(error.item())
                    all_labels.append(labels[i])
        
        # AUC計算
        all_errors = np.array(all_reconstruction_errors)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_errors)
        else:
            auc_score = 0.0
        
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, auc_score
    
    def train_epoch(self, epoch):
        """最適化エポック学習"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_percept_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            
            # 正常データのみで学習（異常検知の基本戦略）
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue
            
            normal_images = images[normal_mask]
            
            # 前向き計算
            self.optimizer.zero_grad()
            reconstructed, latent = self.model(normal_images)
            
            # 損失計算
            total_batch_loss, recon_loss, percept_loss = self.compute_anomaly_loss(normal_images, reconstructed)
            
            # 逆向き計算
            total_batch_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計更新
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_percept_loss += percept_loss.item()
            num_batches += 1
            
            # 進捗表示（5バッチごと）
            if batch_idx % 5 == 0:
                progress = 100. * batch_idx / len(self.train_loader)
                print(f'  Batch [{batch_idx:2d}/{len(self.train_loader):2d}] '
                      f'({progress:4.1f}%) Loss: {total_batch_loss.item():.4f} '
                      f'(R:{recon_loss.item():.4f}, P:{percept_loss.item():.4f})')
        
        # 学習率更新
        self.scheduler.step()
        
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_percept_loss = total_percept_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_total_loss, avg_recon_loss, avg_percept_loss
    
    def train_optimized_full_dataset(self, epochs=25):
        """最適化416枚完全学習"""
        print(f"\n[OPTIMIZED] Full Dataset Anomaly Detection Training")
        print(f"Epochs: {epochs}, Resolution: 512x512, Total samples: 416")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 学習
            train_loss, recon_loss, percept_loss = self.train_epoch(epoch)
            
            # 評価
            val_loss, val_auc = self.evaluate_anomaly_detection(self.val_loader)
            
            # 履歴記録
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_aucs'].append(val_auc)
            
            # ベストモデル更新
            if val_auc > self.training_history['best_auc']:
                self.training_history['best_auc'] = val_auc
                self.training_history['best_epoch'] = epoch
                self.save_model('best')
            
            # エポック結果表示
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Train Loss: {train_loss:.4f} (Recon: {recon_loss:.4f}, Percept: {percept_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            print(f"Best AUC: {self.training_history['best_auc']:.4f} (Epoch {self.training_history['best_epoch']+1})")
            print(f"Learning Rate: {current_lr:.6f}")
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\n[SUCCESS] Optimized Full Dataset Training completed!")
        print(f"Total time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        print(f"Best validation AUC: {self.training_history['best_auc']:.4f}")
        print(f"Final model saved to: models/optimized_full_anomaly/")
        
        # 最終モデル保存
        self.save_model('final')
        self.save_training_history()
        
        return self.training_history
    
    def save_model(self, model_type='final'):
        """最適化モデル保存"""
        models_dir = Path("models/optimized_full_anomaly")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{model_type}_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_size': 512,
                'latent_dim': 256,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
        }, model_path)
        
        print(f"[SAVE] Optimized model saved: {model_path}")
    
    def save_training_history(self):
        """学習履歴保存"""
        models_dir = Path("models/optimized_full_anomaly")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = models_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"[SAVE] History saved: {history_path}")

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='MVTec AD Wallplugs Optimized Full Dataset (416 images)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    print("[OPTIMIZED] MVTec AD Wallplugs Full Dataset Training")
    print("Dataset: 416 images (train 355 + validation 61)")
    print("Resolution: 512x512 (optimized for performance)")
    print("=" * 60)
    
    try:
        trainer = OptimizedWallplugsTrainer(
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # データセット存在確認
        if len(trainer.train_dataset) == 0:
            print("[ERROR] 学習データが見つかりません")
            print("前処理を実行してください: python preprocess_mvtec.py")
            return
        
        # 最適化全データセット学習実行
        history = trainer.train_optimized_full_dataset(epochs=args.epochs)
        
        print(f"\n[SUCCESS] Optimized full dataset training completed!")
        print(f"Final results:")
        print(f"  - Best AUC: {history['best_auc']:.4f}")
        print(f"  - Best Epoch: {history['best_epoch']+1}")
        print(f"  - Total samples: 416 images")
        print(f"  - Resolution: 512x512")
        
        if history['best_auc'] > 0.8:
            print(f"[EXCELLENT] High performance achieved (AUC > 0.8)!")
        elif history['best_auc'] > 0.6:
            print(f"[GOOD] Decent performance achieved (AUC > 0.6)")
        else:
            print(f"[INFO] Performance may need further tuning")
        
    except KeyboardInterrupt:
        print(f"\n[STOP] Training interrupted by user.")
    except Exception as e:
        print(f"\n[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()