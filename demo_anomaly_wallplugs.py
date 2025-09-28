"""
MVTec AD Wallplugs 軽量異常検知モデル学習デモ
MiniCPMを使わない軽量版異常検知モデルの学習テスト
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SimpleWallplugsDataset(Dataset):
    """軽量Wallplugsデータセット"""
    
    def __init__(self, data_dir, split='train', max_samples=50, image_size=256):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size
        
        # サンプル収集
        self.samples = []
        self.collect_samples()
        
        print(f"{split} dataset: {len(self.samples)} samples")
    
    def collect_samples(self):
        """サンプル収集"""
        # 正常画像
        normal_dir = self.data_dir / self.split / "normal"
        if normal_dir.exists():
            normal_files = list(normal_dir.glob("*.png"))[:self.max_samples//2]
            for img_path in normal_files:
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 0  # 0: normal
                })
        
        # 異常画像
        anomalous_dir = self.data_dir / self.split / "anomalous"
        if anomalous_dir.exists():
            anomalous_files = list(anomalous_dir.glob("*.png"))[:self.max_samples//2]
            for img_path in anomalous_files:
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 1  # 1: anomalous
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 画像読み込み・前処理
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # numpy -> tensor
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        return {
            'image': image_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'path': sample['image_path']
        }

class LightweightAutoencoder(nn.Module):
    """軽量オートエンコーダー"""
    
    def __init__(self, input_channels=3, latent_dim=128, image_size=256):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 4, 4)),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(8, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # 異常検知ヘッド
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # エンコード
        latent = self.encoder(x)
        
        # デコード
        reconstructed = self.decoder(latent)
        
        # 異常スコア
        anomaly_score = self.anomaly_head(latent)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'anomaly_score': anomaly_score.squeeze()
        }

class WallplugsLightweightTrainer:
    """軽量異常検知学習クラス"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 出力ディレクトリ
        self.output_dir = Path("models/lightweight_anomaly")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 学習履歴
        self.training_history = {
            'loss': [],
            'reconstruction_loss': [],
            'anomaly_loss': []
        }
    
    def setup_data(self):
        """データ準備"""
        data_dir = Path("data/processed/wallplugs")
        
        if not data_dir.exists():
            print("[ERROR] 前処理済みデータが見つかりません")
            return False
        
        # データセット作成
        self.train_dataset = SimpleWallplugsDataset(
            data_dir, split='train', max_samples=60, image_size=256
        )
        self.val_dataset = SimpleWallplugsDataset(
            data_dir, split='validation', max_samples=20, image_size=256
        )
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2
        )
        
        return True
    
    def setup_model(self):
        """モデル準備"""
        self.model = LightweightAutoencoder(
            input_channels=3,
            latent_dim=128,
            image_size=256
        ).to(self.device)
        
        # オプティマイザー
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10
        )
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def compute_loss(self, output, target, labels):
        """損失計算"""
        reconstructed = output['reconstructed']
        anomaly_scores = output['anomaly_score']
        
        # 再構成損失
        reconstruction_loss = nn.MSELoss()(reconstructed, target)
        
        # 異常検知損失
        anomaly_loss = nn.BCELoss()(anomaly_scores, labels)
        
        # 総損失
        total_loss = reconstruction_loss + 0.5 * anomaly_loss
        
        return total_loss, reconstruction_loss, anomaly_loss
    
    def train_epoch(self, epoch):
        """1エポック学習"""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_anomaly_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 順伝播
            output = self.model(images)
            
            # 損失計算
            loss, recon_loss, anomaly_loss = self.compute_loss(output, images, labels)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            # 損失蓄積
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_anomaly_loss += anomaly_loss.item()
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'Anomaly': f"{anomaly_loss.item():.4f}"
            })
        
        # エポック平均
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_anomaly_loss = total_anomaly_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_anomaly_loss
    
    def validate(self):
        """検証"""
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 順伝播
                output = self.model(images)
                
                # 損失
                loss, _, _ = self.compute_loss(output, images, labels)
                total_loss += loss.item()
                
                # 予測収集
                anomaly_scores = output['anomaly_score'].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                predictions.extend(anomaly_scores)
                ground_truth.extend(labels_np)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # AUC計算
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(ground_truth, predictions)
        except:
            auc_score = 0.0
        
        return avg_loss, auc_score
    
    def train(self, epochs=10):
        """学習実行"""
        print(f"\n[TRAIN] Lightweight Anomaly Detection Training")
        print("="*50)
        
        best_auc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # 学習
            train_loss, train_recon, train_anomaly = self.train_epoch(epoch)
            
            # 検証
            val_loss, val_auc = self.validate()
            
            # 履歴記録
            self.training_history['loss'].append(train_loss)
            self.training_history['reconstruction_loss'].append(train_recon)
            self.training_history['anomaly_loss'].append(train_anomaly)
            
            # ログ出力
            print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, Anomaly: {train_anomaly:.4f})")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # ベストモデル保存
            if val_auc > best_auc:
                best_auc = val_auc
                self.save_model(epoch, val_auc)
                print(f"🎉 New best model! AUC: {val_auc:.4f}")
            
            # スケジューラー更新
            self.scheduler.step()
        
        print(f"\n[SUCCESS] Training completed!")
        print(f"Best validation AUC: {best_auc:.4f}")
        
        return best_auc
    
    def save_model(self, epoch, auc):
        """モデル保存"""
        model_path = self.output_dir / "lightweight_autoencoder_best.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'auc': auc,
            'training_history': self.training_history
        }, model_path)
        
        # 学習履歴保存
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

def main():
    """メイン実行"""
    print("[DEMO] MVTec AD Wallplugs Lightweight Anomaly Detection")
    print("="*60)
    
    try:
        trainer = WallplugsLightweightTrainer()
        
        # データ・モデル準備
        if not trainer.setup_data():
            return False
        
        trainer.setup_model()
        
        # 学習実行
        best_auc = trainer.train(epochs=8)
        
        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Model saved: models/lightweight_anomaly/")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()