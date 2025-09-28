"""
MAD-FH: Logging Utilities
ログ設定用ユーティリティ
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    ロガーの設定
    
    Args:
        name: ロガー名
        log_level: ログレベル
        log_file: ログファイルパス（Noneの場合はコンソールのみ）
        
    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    
    # 既存のハンドラーがある場合は削除
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # フォーマット設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（指定された場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_logger(experiment_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    実験用ロガーの作成
    
    Args:
        experiment_name: 実験名
        log_dir: ログディレクトリ
        
    Returns:
        実験用ロガー
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=f"{experiment_name}_{timestamp}",
        log_file=str(log_file)
    )
