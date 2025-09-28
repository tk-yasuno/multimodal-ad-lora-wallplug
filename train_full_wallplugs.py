"""
MVTec AD Wallplugs 全データセット完全学習システム
train 355枚 + validation 61枚 = 416枚全体対応
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

class WallplugsFullDataset(Dataset):
    """Wallplugs全データセット（416枚対応）"""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: data/processed/wallplugs
            split: 'train' または 'validation'
            transform: 画像変換
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
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
        
        # 画像読み込み
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class FullAnomalyAutoEncoder(nn.Module):
    """全データセット対応異常検知オートエンコーダ"""
    
    def __init__(self, input_size=1024):
        super().__init__()
        
        # エンコーダ（入力: 3x1024x1024 → 潜在表現: 512次元）
        self.encoder = nn.Sequential(
            # 1024 -> 512
            nn.Conv2d(3, 64, 4, 2, 1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 512 -> 256
            nn.Conv2d(64, 128, 4, 2, 1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 256 -> 128
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 128 -> 64
            nn.Conv2d(256, 512, 4, 2, 1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 64 -> 32
            nn.Conv2d(512, 1024, 4, 2, 1), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            nn.Conv2d(1024, 1024, 4, 2, 1), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # 16 -> 8
            nn.Conv2d(1024, 1024, 4, 2, 1), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # 8 -> 4
            nn.Conv2d(1024, 1024, 4, 2, 1), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # 潜在表現変換（4x4x1024 → 512次元）
        self.latent_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 1024, 512),
            nn.ReLU()
        )
        
        # 潜在表現復元（512次元 → 4x4x1024）
        self.latent_restore = nn.Sequential(
            nn.Linear(512, 4 * 4 * 1024),
            nn.ReLU()
        )
        
        # デコーダ（潜在表現: 512次元 → 出力: 3x1024x1024）
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            # 8 -> 16
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            # 16 -> 32
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            # 32 -> 64
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 64 -> 128
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 128 -> 256
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 256 -> 512
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 512 -> 1024
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # エンコード
        encoded = self.encoder(x)
        
        # 潜在表現変換
        latent = self.latent_fc(encoded)
        
        # 潜在表現復元
        restored = self.latent_restore(latent)
        restored = restored.view(-1, 1024, 4, 4)
        
        # デコード
        decoded = self.decoder(restored)
        
        return decoded, latent

class FullWallplugsTrainer:
    """全データセット416枚対応学習クラス"""
    
    def __init__(self, data_dir="data/processed/wallplugs", 
                 batch_size=16, learning_rate=0.0001, device=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # データ変換
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # GPUメモリ効率化設定
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False  # メモリ使用量削減
            torch.backends.cudnn.deterministic = True
            # メモリキャッシュをクリア
            torch.cuda.empty_cache()
        
        # データセット準備
        self.train_dataset = WallplugsFullDataset(
            self.data_dir, split='train', transform=self.transform
        )
        self.val_dataset = WallplugsFullDataset(
            self.data_dir, split='validation', transform=self.transform
        )
        
        # データローダー（メモリ効率化：ワーカー1）
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=1  # メモリ使用量削減
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=1  # メモリ使用量削減
        )
        
        # モデル準備
        self.model = FullAnomalyAutoEncoder().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
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
        print(f"Total parameters: {total_params:,}")
    
    def compute_reconstruction_loss(self, original, reconstructed):
        """再構成損失計算"""
        return self.criterion(reconstructed, original)
    
    def evaluate_anomaly_detection(self, data_loader):
        """異常検知評価"""
        self.model.eval()
        all_losses = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # 前向き計算
                reconstructed, latent = self.model(images)
                
                # 再構成誤差計算（サンプルごと）
                batch_size = images.size(0)
                for i in range(batch_size):
                    img = images[i:i+1]
                    recon = reconstructed[i:i+1]
                    loss = self.compute_reconstruction_loss(img, recon)
                    all_losses.append(loss.item())
                    all_labels.append(labels[i])
        
        # AUC計算
        all_losses = np.array(all_losses)
        all_labels = np.array(all_labels)
        
        auc_score = roc_auc_score(all_labels, all_losses)
        avg_loss = np.mean(all_losses)
        
        return avg_loss, auc_score
    
    def train_epoch(self, epoch):
        """エポック学習"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # 正常データのみで学習（異常検知の基本）
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue  # 正常データがない場合はスキップ
            
            normal_images = images[normal_mask]
            
            # 前向き計算
            self.optimizer.zero_grad()
            reconstructed, latent = self.model(normal_images)
            
            # 損失計算（正常データの再構成のみ）
            loss = self.compute_reconstruction_loss(normal_images, reconstructed)
            
            # 逆向き計算
            loss.backward()
            self.optimizer.step()
            
            # メモリクリア（GPUメモリ最適化）
            del normal_images, reconstructed, latent
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 進捗表示（大きなバッチサイズなので頻度を調整）
            if batch_idx % 2 == 0:
                progress = 100. * batch_idx / len(self.train_loader)
                print(f'  Batch [{batch_idx:2d}/{len(self.train_loader):2d}] '
                      f'({progress:5.1f}%) Loss: {loss.item():.6f} '
                      f'Normal samples in batch: {normal_mask.sum().item()}')
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_train_loss
    
    def train_full_dataset(self, epochs=1):
        """全データセット学習実行（v0-3完成版：1エポック）"""
        print(f"\n[TRAIN] Full Dataset Anomaly Detection Training")
        print(f"Version: v0-3 Final (1 epoch for completion)")
        print(f"Batch size: {self.batch_size} (Memory optimized)")
        print(f"Total samples: 416 (train: 355, val: 61)")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # 学習
            train_loss = self.train_epoch(epoch)
            
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
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.4f}")
            print(f"Best AUC: {self.training_history['best_auc']:.4f} "
                  f"(Epoch {self.training_history['best_epoch']+1})")
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\n[SUCCESS] Full Dataset Training completed!")
        print(f"Version: v0-3 Final completed in {training_time:.1f} seconds")
        print(f"Best validation AUC: {self.training_history['best_auc']:.4f}")
        print(f"v0-3 Final model saved to: models/full_dataset_anomaly/")
        print(f"[ACHIEVEMENT] 416 images complete training finished!")
        
        # 最終モデル保存
        self.save_model('final')
        self.save_training_history()
        
        return self.training_history
    
    def save_model(self, model_type='final'):
        """モデル保存"""
        models_dir = Path("models/full_dataset_anomaly")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル状態保存
        model_path = models_dir / f"{model_type}_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_size': 1024,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
        }, model_path)
        
        print(f"[SAVE] Model saved: {model_path}")
    
    def save_training_history(self):
        """学習履歴保存"""
        models_dir = Path("models/full_dataset_anomaly")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = models_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"[SAVE] History saved: {history_path}")

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='MVTec AD Wallplugs Full Dataset Training (416 images)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (v0-3 final)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    print("[FULL] MVTec AD Wallplugs Full Dataset Training v0-3 Final")
    print("Dataset: 416 images (train 355 + validation 61)")
    print("Strategy: Batch 16, 1 epoch (v0-3 completion)")
    print("=" * 60)
    
    try:
        trainer = FullWallplugsTrainer(
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # データセット存在確認
        if len(trainer.train_dataset) == 0:
            print("[ERROR] 学習データが見つかりません")
            print("前処理を実行してください: python preprocess_mvtec.py")
            return
        
        # 全データセット学習実行
        history = trainer.train_full_dataset(epochs=args.epochs)
        
        print(f"\n[SUCCESS] Full dataset training completed successfully!")
        print(f"Final results:")
        print(f"  - Best AUC: {history['best_auc']:.4f}")
        print(f"  - Best Epoch: {history['best_epoch']+1}")
        print(f"  - Total samples processed: 416 images")
        
    except KeyboardInterrupt:
        print(f"\n[STOP] Training interrupted by user.")
    except Exception as e:
        print(f"\n[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()