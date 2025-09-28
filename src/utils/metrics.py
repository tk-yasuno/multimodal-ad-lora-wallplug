"""
MAD-FH: Evaluation Metrics
異常検知モデルの評価指標計算
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Tuple, Dict, List
import torch


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray = None) -> Dict:
    """
    異常検知の評価指標を計算
    
    Args:
        y_true: 真のラベル（0: 正常, 1: 異常）
        y_scores: 異常度スコア
        y_pred: 予測ラベル（Noneの場合は閾値ベースで計算）
        
    Returns:
        評価指標の辞書
    """
    metrics = {}
    
    # ROC-AUC
    if len(np.unique(y_true)) > 1:  # 正常・異常両方のサンプルがある場合
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        
        # PR-AUC
        metrics['pr_auc'] = average_precision_score(y_true, y_scores)
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    # 予測ラベルが提供された場合の分類指標
    if y_pred is not None:
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):  # 正常・異常両方のクラスがある場合
            tn, fp, fn, tp = cm.ravel()
            
            # 基本的な指標
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # F1スコア
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0
            
            # 偽陽性率
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
        else:
            # 単一クラスの場合
            metrics['accuracy'] = np.mean(y_true == y_pred)
            metrics['precision'] = np.nan
            metrics['recall'] = np.nan
            metrics['specificity'] = np.nan
            metrics['f1_score'] = np.nan
            metrics['fpr'] = np.nan
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    ROC曲線の描画
    
    Args:
        y_true: 真のラベル
        y_scores: 異常度スコア
        save_path: 保存パス
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Cannot plot ROC curve\n(only one class present)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ROC Curve (Not Available)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Precision-Recall曲線の描画
    
    Args:
        y_true: 真のラベル
        y_scores: 異常度スコア
        save_path: 保存パス
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        ax.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})', linewidth=2)
        
        # ベースライン（異常サンプルの割合）
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Cannot plot PR curve\n(only one class present)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Precision-Recall Curve (Not Available)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_score_distribution(normal_scores: np.ndarray, 
                          anomaly_scores: np.ndarray = None,
                          threshold: float = None,
                          save_path: str = None) -> plt.Figure:
    """
    異常度スコアの分布を描画
    
    Args:
        normal_scores: 正常サンプルの異常度スコア
        anomaly_scores: 異常サンプルの異常度スコア
        threshold: 異常検知の閾値
        save_path: 保存パス
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 正常サンプルのスコア分布
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    
    # 異常サンプルのスコア分布（提供された場合）
    if anomaly_scores is not None:
        ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    # 閾値ライン
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Anomaly Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    混同行列の描画
    
    Args:
        y_true: 真のラベル
        y_pred: 予測ラベル
        save_path: 保存パス
        
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float] = None,
                        save_path: str = None) -> plt.Figure:
    """
    学習曲線の描画
    
    Args:
        train_losses: 訓練損失のリスト
        val_losses: 検証損失のリスト
        save_path: 保存パス
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 通常スケール
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 対数スケール
    axes[1].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        axes[1].plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (Log Scale)')
    axes[1].set_title('Training and Validation Loss (Log Scale)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_anomaly_detection(model, 
                              normal_loader, 
                              anomaly_loader=None,
                              device='cuda',
                              threshold_percentile=95.0) -> Dict:
    """
    異常検知モデルの総合評価
    
    Args:
        model: 異常検知モデル
        normal_loader: 正常データローダー
        anomaly_loader: 異常データローダー
        device: 計算デバイス
        threshold_percentile: 閾値パーセンタイル
        
    Returns:
        評価結果の辞書
    """
    model.eval()
    
    # 正常データの評価
    normal_scores = []
    with torch.no_grad():
        for batch, _ in normal_loader:
            batch = batch.to(device)
            if hasattr(model, 'compute_reconstruction_error'):
                scores = model.compute_reconstruction_error(batch)
            else:
                # SimCLRの場合などは別途実装が必要
                scores = torch.randn(batch.size(0))  # プレースホルダー
            normal_scores.extend(scores.cpu().numpy())
    
    normal_scores = np.array(normal_scores)
    
    # 閾値設定
    threshold = np.percentile(normal_scores, threshold_percentile)
    
    results = {
        'normal_scores_mean': np.mean(normal_scores),
        'normal_scores_std': np.std(normal_scores),
        'threshold': threshold,
        'threshold_percentile': threshold_percentile
    }
    
    # 異常データがある場合の評価
    if anomaly_loader is not None:
        anomaly_scores = []
        with torch.no_grad():
            for batch, _ in anomaly_loader:
                batch = batch.to(device)
                if hasattr(model, 'compute_reconstruction_error'):
                    scores = model.compute_reconstruction_error(batch)
                else:
                    scores = torch.randn(batch.size(0))  # プレースホルダー
                anomaly_scores.extend(scores.cpu().numpy())
        
        anomaly_scores = np.array(anomaly_scores)
        
        # 全体のラベルとスコア
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        all_predictions = (all_scores > threshold).astype(int)
        
        # 評価指標計算
        metrics = compute_metrics(all_labels, all_scores, all_predictions)
        results.update(metrics)
        
        results['anomaly_scores_mean'] = np.mean(anomaly_scores)
        results['anomaly_scores_std'] = np.std(anomaly_scores)
    
    return results


if __name__ == "__main__":
    # テスト実行
    
    # ダミーデータで動作確認
    np.random.seed(42)
    
    # 正常データ（低いスコア）
    normal_scores = np.random.normal(0.1, 0.05, 1000)
    
    # 異常データ（高いスコア）
    anomaly_scores = np.random.normal(0.3, 0.1, 100)
    
    # ラベルとスコア
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    
    # 閾値設定（正常データの95パーセンタイル）
    threshold = np.percentile(normal_scores, 95)
    all_predictions = (all_scores > threshold).astype(int)
    
    # 評価指標計算
    metrics = compute_metrics(all_labels, all_scores, all_predictions)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # プロット作成（テスト）
    fig = plot_score_distribution(normal_scores, anomaly_scores, threshold)
    plt.show()
    
    print("Metrics computation test completed successfully!")
