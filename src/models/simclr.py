"""
MAD-FH: SimCLR Model for Anomaly Detection
対照学習ベースの異常検知モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle


class SimCLRProjectionHead(nn.Module):
    """SimCLRのProjection Head"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        """
        Args:
            input_dim: 入力次元数（バックボーンの出力次元）
            hidden_dim: 隠れ層の次元数
            output_dim: 出力次元数
        """
        super(SimCLRProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)


class SimCLRModel(nn.Module):
    """SimCLR モデル"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 projection_dim: int = 128,
                 pretrained: bool = True):
        """
        Args:
            backbone: バックボーンアーキテクチャ
            projection_dim: 射影次元
            pretrained: 事前学習済み重みを使用するか
        """
        super(SimCLRModel, self).__init__()
        
        # バックボーンの選択
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            # 最後の全結合層を除去
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection Head
        self.projection_head = SimCLRProjectionHead(backbone_dim, output_dim=projection_dim)
        
        self.backbone_dim = backbone_dim
        self.projection_dim = projection_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 入力画像テンソル [B, C, H, W]
            
        Returns:
            features: バックボーンの特徴量 [B, backbone_dim]
            projections: 射影された特徴量 [B, projection_dim]
        """
        # バックボーンで特徴抽出
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # 射影
        projections = self.projection_head(features)
        
        return features, projections
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """バックボーンの特徴量のみを取得"""
        with torch.no_grad():
            features = self.backbone(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
        return features
    
    def get_projections(self, x: torch.Tensor) -> torch.Tensor:
        """射影特徴量のみを取得"""
        with torch.no_grad():
            _, projections = self.forward(x)
        return projections


class InfoNCELoss(nn.Module):
    """InfoNCE損失関数"""
    
    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: 温度パラメータ
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE損失の計算
        
        Args:
            projections: 射影された特徴量 [2*B, projection_dim]
                        (各画像のaugmented pairが連続で配置されている前提)
        
        Returns:
            loss: InfoNCE損失
        """
        batch_size = projections.size(0) // 2
        device = projections.device
        
        # 正例のペアのインデックス
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        
        # 類似度行列の計算
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # 対角要素をマスク（自分自身との類似度を除外）
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 正例と負例の分離
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # InfoNCE損失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SimCLRAnomalyDetector:
    """SimCLRベースの異常検知器"""
    
    def __init__(self, 
                 model: SimCLRModel,
                 device: str = 'cuda',
                 k_neighbors: int = 5):
        """
        Args:
            model: 学習済みSimCLRモデル
            device: 計算デバイス
            k_neighbors: k-NN探索での近傍数
        """
        self.model = model
        self.device = device
        self.k_neighbors = k_neighbors
        
        self.model.to(device)
        self.model.eval()
        
        # 正常データの特徴量データベース
        self.normal_features = None
        self.knn_model = None
        self.threshold = None
    
    def fit(self, normal_data_loader, use_projections: bool = False):
        """
        正常データから特徴量データベースを構築
        
        Args:
            normal_data_loader: 正常データのDataLoader
            use_projections: 射影特徴量を使用するか（Falseの場合はバックボーン特徴量）
        """
        features_list = []
        
        with torch.no_grad():
            for batch, _ in normal_data_loader:
                batch = batch.to(self.device)
                
                if use_projections:
                    projections = self.model.get_projections(batch)
                    features_list.append(projections.cpu().numpy())
                else:
                    features = self.model.get_features(batch)
                    features_list.append(features.cpu().numpy())
        
        # 特徴量を結合
        self.normal_features = np.vstack(features_list)
        
        # k-NNモデルの構築
        self.knn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric='euclidean',
            algorithm='auto'
        )
        self.knn_model.fit(self.normal_features)
        
        print(f"Normal features database built: {self.normal_features.shape}")
    
    def fit_threshold(self, normal_data_loader, percentile: float = 95.0, use_projections: bool = False):
        """
        正常データから異常度の閾値を設定
        
        Args:
            normal_data_loader: 正常データのDataLoader
            percentile: 閾値パーセンタイル
            use_projections: 射影特徴量を使用するか
        """
        if self.knn_model is None:
            raise ValueError("Please call fit() first to build the feature database.")
        
        distances_list = []
        
        with torch.no_grad():
            for batch, _ in normal_data_loader:
                batch = batch.to(self.device)
                
                if use_projections:
                    features = self.model.get_projections(batch).cpu().numpy()
                else:
                    features = self.model.get_features(batch).cpu().numpy()
                
                # k-NN距離を計算
                distances, _ = self.knn_model.kneighbors(features)
                mean_distances = np.mean(distances, axis=1)
                distances_list.extend(mean_distances)
        
        distances_array = np.array(distances_list)
        self.threshold = np.percentile(distances_array, percentile)
        
        print(f"Anomaly threshold (>{percentile}%): {self.threshold:.4f}")
    
    def predict(self, x: torch.Tensor, use_projections: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        異常検知の実行
        
        Args:
            x: 入力画像テンソル
            use_projections: 射影特徴量を使用するか
            
        Returns:
            scores: 異常度スコア（k-NN平均距離）
            predictions: 異常判定（1: 異常, 0: 正常）
        """
        if self.knn_model is None:
            raise ValueError("Please call fit() first to build the feature database.")
        
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            
            if use_projections:
                features = self.model.get_projections(x).cpu().numpy()
            else:
                features = self.model.get_features(x).cpu().numpy()
        
        # k-NN距離を計算
        distances, indices = self.knn_model.kneighbors(features)
        scores = np.mean(distances, axis=1)
        
        # 異常判定
        if self.threshold is not None:
            predictions = (scores > self.threshold).astype(int)
        else:
            predictions = np.zeros_like(scores, dtype=int)
        
        return scores, predictions
    
    def get_nearest_neighbors(self, x: torch.Tensor, use_projections: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        最近傍サンプルの取得
        
        Returns:
            distances: 最近傍距離
            indices: 最近傍インデックス
        """
        if self.knn_model is None:
            raise ValueError("Please call fit() first to build the feature database.")
        
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            
            if use_projections:
                features = self.model.get_projections(x).cpu().numpy()
            else:
                features = self.model.get_features(x).cpu().numpy()
        
        distances, indices = self.knn_model.kneighbors(features)
        return distances, indices
    
    def save_feature_database(self, filepath: str):
        """特徴量データベースの保存"""
        data = {
            'normal_features': self.normal_features,
            'knn_model': self.knn_model,
            'threshold': self.threshold,
            'k_neighbors': self.k_neighbors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_feature_database(self, filepath: str):
        """特徴量データベースの読み込み"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.normal_features = data['normal_features']
        self.knn_model = data['knn_model']
        self.threshold = data['threshold']
        self.k_neighbors = data['k_neighbors']


# Augmentationヘルパー関数
def create_simclr_augmentation():
    """SimCLR用のAugmentation"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(height=224, width=224),  # SimCLRは通常224x224
        A.OneOf([
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
            A.CenterCrop(height=224, width=224),
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            A.ToGray(p=0.2),
        ], p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return transform


if __name__ == "__main__":
    # テスト実行
    
    # モデル初期化
    model = SimCLRModel(backbone='resnet50', projection_dim=128)
    
    # サンプル入力で動作確認
    batch_size = 4
    sample_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {sample_input.shape}")
    
    # 順伝播
    features, projections = model(sample_input)
    print(f"Features shape: {features.shape}")
    print(f"Projections shape: {projections.shape}")
    
    # InfoNCE損失のテスト
    loss_fn = InfoNCELoss(temperature=0.1)
    
    # Augmentedペアをシミュレート
    augmented_batch = torch.randn(batch_size * 2, 3, 224, 224)
    _, aug_projections = model(augmented_batch)
    
    loss = loss_fn(aug_projections)
    print(f"InfoNCE loss: {loss.item():.4f}")
    
    # 異常検知器のテスト
    detector = SimCLRAnomalyDetector(model, device='cpu', k_neighbors=3)
    
    # ダミーの正常データでフィット
    dummy_normal = torch.randn(10, 3, 224, 224)
    dummy_loader = [(dummy_normal[i:i+2], None) for i in range(0, 10, 2)]
    
    detector.fit(dummy_loader)
    detector.fit_threshold(dummy_loader, percentile=90.0)
    
    # 予測テスト
    test_input = torch.randn(2, 3, 224, 224)
    scores, predictions = detector.predict(test_input)
    print(f"Anomaly scores: {scores}")
    print(f"Predictions: {predictions}")
    
    print("SimCLR model test completed successfully!")
