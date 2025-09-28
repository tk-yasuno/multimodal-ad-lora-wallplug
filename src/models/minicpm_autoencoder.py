"""
MAD-FH: MiniCPM-enhanced Autoencoder Model for Anomaly Detection
MiniCPM言語モデルを統合した高度な異常検知システム
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class MiniCPMVisionEncoder(nn.Module):
    """MiniCPMビジョンエンコーダー"""
    
    def __init__(self, model_name="openbmb/MiniCPM-V-2_6", freeze_backbone=True):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        try:
            # MiniCPMモデルの読み込み
            self.vision_model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # 画像プロセッサー
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 特徴量次元を取得
            self.feature_dim = self._get_feature_dim()
            
            if freeze_backbone:
                # バックボーンを固定
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                    
            logger.info(f"MiniCPM Vision Encoder loaded: {model_name}")
            logger.info(f"Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load MiniCPM model: {e}")
            # フォールバックとして基本CNN
            self.vision_model = self._create_fallback_cnn()
            self.feature_dim = 2048
            self.image_processor = None
            
    def _get_feature_dim(self):
        """特徴量次元を動的に取得"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(self.vision_model, 'get_vision_hidden_states'):
                features = self.vision_model.get_vision_hidden_states(dummy_input)
                return features.shape[-1]
            else:
                return 4096  # デフォルト値
        except:
            return 4096
    
    def _create_fallback_cnn(self):
        """フォールバック用CNN"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 2048)
        )
    
    def preprocess_image(self, image):
        """画像の前処理"""
        if self.image_processor is not None:
            if isinstance(image, torch.Tensor):
                # Tensor to PIL
                image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            
            inputs = self.image_processor(image, return_tensors="pt")
            return inputs.pixel_values.to(next(self.parameters()).device)
        else:
            # フォールバック処理
            if isinstance(image, Image.Image):
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def forward(self, x):
        """順伝播"""
        try:
            if hasattr(self.vision_model, 'get_vision_hidden_states'):
                # MiniCPM vision features
                features = self.vision_model.get_vision_hidden_states(x)
                return features.mean(dim=1)  # [batch, seq, hidden] -> [batch, hidden]
            else:
                # フォールバック
                return self.vision_model(x)
        except Exception as e:
            logger.warning(f"Vision encoding error: {e}, using fallback")
            return self.vision_model(x)

class MiniCPMHybridAutoencoder(nn.Module):
    """MiniCPM統合ハイブリッド異常検知オートエンコーダー"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 512,
                 input_size: Tuple[int, int] = (1024, 1024),
                 use_minicpm: bool = True,
                 minicpm_weight: float = 0.3):
        """
        Args:
            input_channels: 入力チャンネル数
            latent_dim: 潜在空間次元
            input_size: 入力画像サイズ
            use_minicpm: MiniCPM使用フラグ
            minicpm_weight: MiniCPM特徴量の重み
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.use_minicpm = use_minicpm
        self.minicpm_weight = minicpm_weight
        
        # 従来のCNN Encoder
        self.cnn_encoder = self._build_cnn_encoder()
        
        # MiniCPM Vision Encoder
        if use_minicpm:
            self.minicpm_encoder = MiniCPMVisionEncoder()
            self.minicpm_projector = nn.Linear(
                self.minicpm_encoder.feature_dim, 
                latent_dim
            )
        
        # 特徴融合層
        if use_minicpm:
            self.feature_fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim, latent_dim)
            )
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # 異常検知ヘッド
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"MiniCPM Hybrid Autoencoder initialized")
        logger.info(f"MiniCPM enabled: {use_minicpm}")
        
    def _build_cnn_encoder(self):
        """CNN エンコーダー構築"""
        # 1024x1024 対応のエンコーダー
        return nn.Sequential(
            # 1024x1024 -> 512x512
            nn.Conv2d(self.input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 512x512 -> 256x256
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 128x128
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, self.latent_dim)
        )
    
    def _build_decoder(self):
        """デコーダー構築"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 1024 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (1024, 4, 4)),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 512x512 -> 1024x1024
            nn.ConvTranspose2d(8, self.input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """エンコード処理"""
        # CNN特徴量抽出
        cnn_features = self.cnn_encoder(x)
        
        if self.use_minicpm:
            try:
                # MiniCPM特徴量抽出
                # 画像サイズをMiniCPM用に調整 (1024x1024 -> 224x224)
                x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                minicpm_features = self.minicpm_encoder(x_resized)
                minicpm_features = self.minicpm_projector(minicpm_features)
                
                # 特徴量融合
                combined_features = torch.cat([cnn_features, minicpm_features], dim=1)
                fused_features = self.feature_fusion(combined_features)
                
                return fused_features
                
            except Exception as e:
                logger.warning(f"MiniCPM encoding failed: {e}, using CNN only")
                return cnn_features
        else:
            return cnn_features
    
    def decode(self, z):
        """デコード処理"""
        return self.decoder(z)
    
    def forward(self, x):
        """順伝播"""
        # エンコード
        latent = self.encode(x)
        
        # デコード
        reconstructed = self.decode(latent)
        
        # 異常スコア計算
        anomaly_score = self.anomaly_head(latent)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'anomaly_score': anomaly_score.squeeze()
        }
    
    def compute_loss(self, x, output, anomaly_labels=None):
        """損失計算"""
        reconstructed = output['reconstructed']
        anomaly_scores = output['anomaly_score']
        
        # 再構成損失 (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # 知覚損失 (L1)
        perceptual_loss = F.l1_loss(reconstructed, x, reduction='mean')
        
        # 異常検知損失（ラベルがある場合）
        if anomaly_labels is not None:
            anomaly_loss = F.binary_cross_entropy(
                anomaly_scores, 
                anomaly_labels.float(), 
                reduction='mean'
            )
        else:
            # tensorとして0を作成（device対応）
            anomaly_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # 総損失
        total_loss = (
            reconstruction_loss + 
            0.1 * perceptual_loss + 
            0.3 * anomaly_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'perceptual_loss': perceptual_loss,
            'anomaly_loss': anomaly_loss
        }

class MiniCPMAnomalyDetector:
    """MiniCPM統合異常検知システム"""
    
    def __init__(self, 
                 model_config: Dict,
                 device: Optional[str] = None):
        """
        Args:
            model_config: モデル設定
            device: デバイス指定
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = model_config
        
        # モデル初期化
        self.model = MiniCPMHybridAutoencoder(
            input_channels=model_config.get('input_channels', 3),
            latent_dim=model_config.get('latent_dim', 512),
            input_size=model_config.get('input_size', (1024, 1024)),
            use_minicpm=model_config.get('use_minicpm', True),
            minicpm_weight=model_config.get('minicpm_weight', 0.3)
        ).to(self.device)
        
        # 閾値
        self.anomaly_threshold = model_config.get('anomaly_threshold', 0.1)
        
        logger.info(f"MiniCPM Anomaly Detector initialized on {self.device}")
    
    def train_step(self, batch_data):
        """1ステップの訓練"""
        self.model.train()
        
        images = batch_data['images'].to(self.device)
        anomaly_labels = batch_data.get('anomaly_labels')
        if anomaly_labels is not None:
            anomaly_labels = anomaly_labels.to(self.device)
        
        # 順伝播
        output = self.model(images)
        
        # 損失計算
        losses = self.model.compute_loss(images, output, anomaly_labels)
        
        return losses, output
    
    def predict(self, image):
        """異常検知予測"""
        self.model.eval()
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            
            # 再構成誤差
            reconstruction_error = F.mse_loss(
                output['reconstructed'], 
                image, 
                reduction='none'
            ).mean(dim=[1, 2, 3])
            
            # 異常スコア（モデル出力 + 再構成誤差）
            model_anomaly_score = output['anomaly_score']
            combined_score = (
                0.7 * model_anomaly_score.cpu().numpy() + 
                0.3 * reconstruction_error.cpu().numpy()
            )
            
            # 異常判定
            is_anomaly = combined_score > self.anomaly_threshold
            
            return {
                'is_anomaly': bool(is_anomaly[0]),
                'anomaly_score': float(combined_score[0]),
                'reconstruction_error': float(reconstruction_error[0]),
                'model_score': float(model_anomaly_score[0]),
                'confidence': float(abs(combined_score[0] - self.anomaly_threshold) * 2)
            }
    
    def save_model(self, filepath):
        """モデル保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'anomaly_threshold': self.anomaly_threshold
        }, filepath)
        logger.info(f"Model saved: {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=device)
        
        detector = cls(
            model_config=checkpoint['config'],
            device=device
        )
        
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        detector.anomaly_threshold = checkpoint.get('anomaly_threshold', 0.1)
        
        logger.info(f"Model loaded: {filepath}")
        return detector

# 元のクラスとの互換性のためのエイリアス
ConvAutoencoder = MiniCPMHybridAutoencoder
AnomalyDetector = MiniCPMAnomalyDetector