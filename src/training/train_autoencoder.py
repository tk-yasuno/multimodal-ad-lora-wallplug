"""
MAD-FH: Training Script for Autoencoder
Autoencoderモデルの学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

# プロジェクト内のモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import ConvAutoencoder, VariationalAutoencoder, AnomalyDetector
from data.preprocess import NormalImageDataset, ImagePreprocessor
from utils.logger import setup_logger
from utils.metrics import compute_metrics, plot_training_curves


class AutoencoderTrainer:
    """Autoencoder学習クラス"""
    
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
        self.logger = setup_logger("autoencoder_training")
        
        # 学習履歴
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # モデル保存ディレクトリ
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 結果保存ディレクトリ
        self.results_dir = Path("results") / f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model(self, model_type: str = 'autoencoder') -> nn.Module:
        """
        モデルの作成
        
        Args:
            model_type: 'autoencoder' または 'vae'
            
        Returns:
            作成されたモデル
        """
        model_config = self.config['models']['autoencoder']
        
        if model_type == 'autoencoder':
            model = ConvAutoencoder(
                input_channels=model_config['input_channels'],
                latent_dim=model_config['latent_dim'],
                input_size=(512, 512)  # 設定から取得可能
            )
        elif model_type == 'vae':
            model = VariationalAutoencoder(
                input_channels=model_config['input_channels'],
                latent_dim=model_config['latent_dim'],
                input_size=(512, 512)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def create_data_loaders(self, processed_images_dir: str) -> tuple:
        """
        DataLoaderの作成
        
        Args:
            processed_images_dir: 前処理済み画像ディレクトリ
            
        Returns:
            train_loader, val_loader
        """
        # 前処理済み画像ファイルの取得
        processed_dir = Path(processed_images_dir)
        image_files = list(processed_dir.glob("*.pt"))
        
        if not image_files:
            raise FileNotFoundError(f"No processed images found in {processed_images_dir}")
        
        # 画像パスのリスト（.ptファイルから元画像パスを推定）
        image_paths = []
        for pt_file in image_files:
            # .ptファイルからテンソルを読み込んで、元の画像パスを推定
            # ここでは簡略化して.ptファイルのパスをそのまま使用
            image_paths.append(str(pt_file))
        
        # 前処理器の初期化
        preprocessor = ImagePreprocessor()
        
        # データセットの作成
        dataset = NormalImageDataset(
            image_paths=image_paths,
            preprocessor=preprocessor,
            apply_augmentation=True  # 学習時はAugmentation適用
        )
        
        # 訓練・検証分割
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # DataLoaderの作成
        batch_size = self.config['models']['autoencoder']['batch_size']
        num_workers = self.config['training']['num_workers']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        self.logger.info(f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, model_type: str = 'autoencoder') -> float:
        """
        1エポックの学習
        
        Args:
            model: 学習対象モデル
            train_loader: 訓練データローダー
            optimizer: オプティマイザー
            model_type: モデルタイプ
            
        Returns:
            平均損失
        """
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc="Training")):
            data = data.to(self.device)
            
            optimizer.zero_grad()
            
            if model_type == 'vae':
                # VAEの場合
                reconstructed, mu, logvar = model(data)
                losses = model.compute_vae_loss(data, reconstructed, mu, logvar)
                loss = losses['total_loss']
            else:
                # 通常のAutoencoderの場合
                reconstructed, _ = model(data)
                loss = nn.MSELoss()(reconstructed, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, model_type: str = 'autoencoder') -> float:
        """
        1エポックの検証
        
        Args:
            model: 検証対象モデル
            val_loader: 検証データローダー
            model_type: モデルタイプ
            
        Returns:
            平均損失
        """
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)
                
                if model_type == 'vae':
                    reconstructed, mu, logvar = model(data)
                    losses = model.compute_vae_loss(data, reconstructed, mu, logvar)
                    loss = losses['total_loss']
                else:
                    reconstructed, _ = model(data)
                    loss = nn.MSELoss()(reconstructed, data)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, processed_images_dir: str, model_type: str = 'autoencoder'):
        """
        学習の実行
        
        Args:
            processed_images_dir: 前処理済み画像ディレクトリ
            model_type: モデルタイプ
        """
        # データローダーの作成
        train_loader, val_loader = self.create_data_loaders(processed_images_dir)
        
        # モデルの作成
        model = self.create_model(model_type)
        
        # オプティマイザーの設定
        learning_rate = self.config['models']['autoencoder']['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学習ループ
        epochs = self.config['models']['autoencoder']['epochs']
        checkpoint_interval = self.config['training']['checkpoint_interval']
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # 学習
            train_loss = self.train_epoch(model, train_loader, optimizer, model_type)
            
            # 検証
            val_loss = self.validate_epoch(model, val_loader, model_type)
            
            # 履歴の記録
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # ベストモデルの保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = self.model_dir / f"best_{model_type}_model.pt"
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved: {best_model_path}")
            
            # 定期的なチェックポイント保存
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = self.model_dir / f"{model_type}_checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
        
        # 最終モデルの保存
        final_model_path = self.model_dir / f"final_{model_type}_model.pt"
        torch.save(model.state_dict(), final_model_path)
        
        # 学習曲線の描画
        self.plot_training_curves()
        
        # 学習結果の保存
        self.save_training_results(model_type)
        
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        
        return model
    
    def plot_training_curves(self):
        """学習曲線の描画"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.log(self.train_losses), label='Log Train Loss')
        plt.plot(np.log(self.val_losses), label='Log Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_results(self, model_type: str):
        """学習結果の保存"""
        results = {
            'model_type': model_type,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'device': self.device
        }
        
        with open(self.results_dir / "training_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Train Autoencoder for Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--processed_dir', type=str, default='data/processed_images', help='Processed images directory')
    parser.add_argument('--model_type', type=str, choices=['autoencoder', 'vae'], default='autoencoder', help='Model type')
    
    args = parser.parse_args()
    
    # 学習器の初期化
    trainer = AutoencoderTrainer(args.config)
    
    # 学習実行
    model = trainer.train(args.processed_dir, args.model_type)
    
    print(f"Training completed successfully!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Results saved to: {trainer.results_dir}")


if __name__ == "__main__":
    main()
