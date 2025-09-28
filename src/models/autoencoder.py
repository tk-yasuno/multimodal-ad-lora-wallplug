"""
MAD-FH: Autoencoder Model for Anomaly Detection
再構成誤差ベースの異常検知モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class ConvAutoencoder(nn.Module):
    """畳み込みAutoencoder"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 input_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            input_channels: 入力チャンネル数
            latent_dim: 潜在空間の次元数
            input_size: 入力画像サイズ (H, W)
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 128x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 潜在空間への変換
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # 潜在空間から特徴マップへの復元
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 16 * 16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 出力を[0,1]に正規化
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """エンコード"""
        x = self.encoder(x)
        x = self.encoder_fc(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """デコード"""
        x = self.decoder_fc(z)
        x = x.view(-1, 512, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 入力画像テンソル [B, C, H, W]
            
        Returns:
            reconstructed: 再構成画像 [B, C, H, W]
            latent: 潜在表現 [B, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def compute_reconstruction_error(self, 
                                   x: torch.Tensor, 
                                   reduction: str = 'mean') -> torch.Tensor:
        """
        再構成誤差の計算
        
        Args:
            x: 入力画像テンソル
            reduction: 'mean', 'sum', 'none'
            
        Returns:
            再構成誤差
        """
        reconstructed, _ = self.forward(x)
        
        # ピクセル単位のMSE
        mse = F.mse_loss(reconstructed, x, reduction='none')
        
        if reduction == 'mean':
            return mse.view(mse.size(0), -1).mean(dim=1)
        elif reduction == 'sum':
            return mse.view(mse.size(0), -1).sum(dim=1)
        else:
            return mse


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE)"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 input_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            input_channels: 入力チャンネル数
            latent_dim: 潜在空間の次元数
            input_size: 入力画像サイズ (H, W)
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # エンコーダー（ConvAutoencoderと同じ構造）
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 平均と分散の推定
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)
        
        # デコーダー
        self.decoder_fc = nn.Linear(latent_dim, 512 * 16 * 16)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコード"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """デコード"""
        x = F.relu(self.decoder_fc(z))
        x = x.view(-1, 512, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Returns:
            reconstructed: 再構成画像
            mu: 潜在分布の平均
            logvar: 潜在分布の対数分散
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_vae_loss(self, 
                        x: torch.Tensor, 
                        reconstructed: torch.Tensor,
                        mu: torch.Tensor, 
                        logvar: torch.Tensor,
                        beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        VAE損失の計算
        
        Args:
            x: 入力画像
            reconstructed: 再構成画像
            mu: 潜在分布の平均
            logvar: 潜在分布の対数分散
            beta: KL項の重み（β-VAE）
            
        Returns:
            損失値の辞書
        """
        # 再構成損失
        recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
        
        # KL発散
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 総損失
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }


class AnomalyDetector:
    """Autoencoderベースの異常検知器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 学習済みのAutoencoderモデル
            device: 計算デバイス
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 正常データの統計情報
        self.normal_scores_mean = None
        self.normal_scores_std = None
        self.threshold = None
    
    def fit_threshold(self, normal_data_loader, percentile: float = 95.0):
        """
        正常データから異常度の閾値を設定
        
        Args:
            normal_data_loader: 正常データのDataLoader
            percentile: 閾値パーセンタイル（95.0なら上位5%を異常とする）
        """
        scores = []
        
        with torch.no_grad():
            for batch, _ in normal_data_loader:
                batch = batch.to(self.device)
                reconstruction_errors = self.model.compute_reconstruction_error(batch)
                scores.extend(reconstruction_errors.cpu().numpy())
        
        scores = np.array(scores)
        self.normal_scores_mean = np.mean(scores)
        self.normal_scores_std = np.std(scores)
        self.threshold = np.percentile(scores, percentile)
        
        print(f"Normal scores - Mean: {self.normal_scores_mean:.4f}, Std: {self.normal_scores_std:.4f}")
        print(f"Anomaly threshold (>{percentile}%): {self.threshold:.4f}")
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        異常検知の実行
        
        Args:
            x: 入力画像テンソル
            
        Returns:
            scores: 異常度スコア
            predictions: 異常判定（1: 異常, 0: 正常）
        """
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            scores = self.model.compute_reconstruction_error(x)
            
            if self.threshold is not None:
                predictions = (scores > self.threshold).long()
            else:
                predictions = torch.zeros_like(scores, dtype=torch.long)
        
        return scores.cpu(), predictions.cpu()
    
    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """再構成画像を取得"""
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            if isinstance(self.model, VariationalAutoencoder):
                reconstructed, _, _ = self.model(x)
            else:
                reconstructed, _ = self.model(x)
        
        return reconstructed.cpu()


if __name__ == "__main__":
    # テスト実行
    
    # モデル初期化
    model = ConvAutoencoder(input_channels=3, latent_dim=256, input_size=(512, 512))
    
    # サンプル入力で動作確認
    batch_size = 4
    sample_input = torch.randn(batch_size, 3, 512, 512)
    
    print(f"Input shape: {sample_input.shape}")
    
    # 順伝播
    reconstructed, latent = model(sample_input)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # 再構成誤差計算
    recon_error = model.compute_reconstruction_error(sample_input)
    print(f"Reconstruction error shape: {recon_error.shape}")
    print(f"Reconstruction errors: {recon_error}")
    
    # VAEテスト
    vae_model = VariationalAutoencoder(input_channels=3, latent_dim=256)
    reconstructed, mu, logvar = vae_model(sample_input)
    
    losses = vae_model.compute_vae_loss(sample_input, reconstructed, mu, logvar)
    print(f"VAE losses: {losses}")
    
    print("Autoencoder models test completed successfully!")
