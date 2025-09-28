"""
MAD-FH: Training Script for SimCLR
SimCLRモデルの学習スクリプト
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# プロジェクト内のモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.simclr import SimCLRModel, InfoNCELoss, create_simclr_augmentation
from data.preprocess import ImagePreprocessor
from utils.logger import setup_logger


class SimCLRDataset(Dataset):
    """SimCLR用のデータセット（Augmentedペア生成）"""
    
    def __init__(self, image_paths: list, transform):
        """
        Args:
            image_paths: 画像パスのリスト
            transform: Augmentation変換
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # 画像読み込み
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2つの異なるAugmentationを適用
            augmented1 = self.transform(image=image)['image']
            augmented2 = self.transform(image=image)['image']
            
            return augmented1, augmented2, image_path
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # エラー時はゼロテンソルを返す
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, dummy_tensor, image_path


class SimCLRTrainer:
    """SimCLR学習クラス"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # デバイス設定
        if self.config['training']['device'] == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.config['training']['device']
        
        print(f"Using device: {self.device}")
        
        # ログの設定
        self.logger = setup_logger("simclr_training")
        
        # 学習履歴
        self.train_losses = []
        self.best_loss = float('inf')
        
        # モデル保存ディレクトリ
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 結果保存ディレクトリ
        self.results_dir = Path("results") / f"simclr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model(self) -> SimCLRModel:
        """モデルの作成"""
        model_config = self.config['models']['simclr']
        
        model = SimCLRModel(
            backbone=model_config['backbone'],
            projection_dim=model_config['projection_dim'],
            pretrained=True
        )
        
        return model.to(self.device)
    
    def create_data_loader(self, image_directory: str) -> DataLoader:
        """
        DataLoaderの作成
        
        Args:
            image_directory: 画像ディレクトリ
            
        Returns:
            データローダー
        """
        # 画像ファイルの取得
        image_dir = Path(image_directory)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(image_dir.rglob(f"*{ext}")))
            image_paths.extend(list(image_dir.rglob(f"*{ext.upper()}")))
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_directory}")
        
        # 文字列パスに変換
        image_paths = [str(path) for path in image_paths]
        
        # Augmentation変換の作成
        transform = create_simclr_augmentation()
        
        # データセットの作成
        dataset = SimCLRDataset(image_paths, transform)
        
        # DataLoaderの作成
        batch_size = self.config['models']['simclr']['batch_size']
        num_workers = self.config['training']['num_workers']
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True  # バッチサイズを一定に保つ
        )
        
        self.logger.info(f"Dataset created - Total images: {len(image_paths)}")
        
        return data_loader
    
    def train_epoch(self, model: SimCLRModel, data_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: InfoNCELoss) -> float:
        """
        1エポックの学習
        
        Args:
            model: SimCLRモデル
            data_loader: データローダー
            optimizer: オプティマイザー
            loss_fn: 損失関数
            
        Returns:
            平均損失
        """
        model.train()
        total_loss = 0
        
        for batch_idx, (img1, img2, _) in enumerate(tqdm(data_loader, desc="Training")):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # バッチを結合（img1とimg2を交互に配置）
            batch = torch.cat([img1, img2], dim=0)
            
            optimizer.zero_grad()
            
            # 順伝播
            _, projections = model(batch)
            
            # InfoNCE損失計算
            loss = loss_fn(projections)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def train(self, image_directory: str):
        """
        学習の実行
        
        Args:
            image_directory: 画像ディレクトリ
        """
        # データローダーの作成
        data_loader = self.create_data_loader(image_directory)
        
        # モデルの作成
        model = self.create_model()
        
        # 損失関数とオプティマイザーの設定
        model_config = self.config['models']['simclr']
        loss_fn = InfoNCELoss(temperature=model_config['temperature'])
        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
        
        # 学習スケジューラー
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_config['epochs'], eta_min=1e-6
        )
        
        # 学習ループ
        epochs = model_config['epochs']
        checkpoint_interval = self.config['training']['checkpoint_interval']
        
        self.logger.info(f"Starting SimCLR training for {epochs} epochs")
        
        for epoch in range(epochs):
            # 学習
            train_loss = self.train_epoch(model, data_loader, optimizer, loss_fn)
            
            # 学習率更新
            scheduler.step()
            
            # 履歴の記録
            self.train_losses.append(train_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}, LR: {current_lr:.2e}")
            
            # ベストモデルの保存
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                best_model_path = self.model_dir / "best_simclr_model.pt"
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved: {best_model_path}")
            
            # 定期的なチェックポイント保存
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = self.model_dir / f"simclr_checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                }, checkpoint_path)
        
        # 最終モデルの保存
        final_model_path = self.model_dir / "final_simclr_model.pt"
        torch.save(model.state_dict(), final_model_path)
        
        # 学習曲線の描画
        self.plot_training_curves()
        
        # 学習結果の保存
        self.save_training_results()
        
        self.logger.info(f"Training completed. Best loss: {self.best_loss:.6f}")
        
        return model
    
    def plot_training_curves(self):
        """学習曲線の描画"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('InfoNCE Loss')
        plt.title('SimCLR Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.log(self.train_losses), label='Log Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('SimCLR Training Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "simclr_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_results(self):
        """学習結果の保存"""
        results = {
            'model_type': 'simclr',
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'total_epochs': len(self.train_losses),
            'device': self.device
        }
        
        with open(self.results_dir / "simclr_training_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR for Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--image_dir', type=str, default='data/images/normal', help='Image directory')
    
    args = parser.parse_args()
    
    # 学習器の初期化
    trainer = SimCLRTrainer(args.config)
    
    # 学習実行
    model = trainer.train(args.image_dir)
    
    print(f"SimCLR training completed successfully!")
    print(f"Best loss: {trainer.best_loss:.6f}")
    print(f"Results saved to: {trainer.results_dir}")


if __name__ == "__main__":
    main()
