#!/usr/bin/env python3
"""
MAD-FH LoRAテキスト生成モデル学習スクリプト

使用方法:
    python train_lora_model.py [--config CONFIG_PATH] [--demo]

オプション:
    --config: 設定ファイルのパス (デフォルト: config/config.yaml)
    --demo: デモモード（軽量学習設定）
    --force-dummy: ダミーデータを強制使用
"""

import argparse
import logging
import sys
import os
import json
import glob
import random
from pathlib import Path
import torch
import yaml
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    """ログ設定"""
    # ログディレクトリの作成
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'lora_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file))
        ]
    )

def check_environment():
    """環境チェック"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Python バージョン: {sys.version}")
    logger.info(f"PyTorch バージョン: {torch.__version__}")
    logger.info(f"CUDA 利用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA デバイス: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDAが利用できません。CPUで学習します（非常に遅くなります）")
    
    # 必要なディレクトリの作成
    for directory in ['logs', 'models/lora', 'data/text_generation']:
        Path(directory).mkdir(parents=True, exist_ok=True)

def prepare_dummy_data(config):
    """実際の画像ファイルを使用したダミーデータの作成"""
    logger = logging.getLogger(__name__)
    
    logger.info("実際の画像を使用したダミー学習データを作成中...")
    
    # 実際の画像ファイルパスを取得
    import glob
    import random
    
    metal_plate_images = glob.glob("data/mpdd/raw/metal_plate/train/good/*.png")
    tube_images = glob.glob("data/mpdd/raw/tubes/train/good/*.png")
    
    # 各カテゴリから数枚選択
    selected_metal = random.sample(metal_plate_images, min(4, len(metal_plate_images)))
    selected_tubes = random.sample(tube_images, min(4, len(tube_images)))
    
    dummy_data = []
    
    # 金属プレート画像用のダミーデータ
    for i, image_path in enumerate(selected_metal):
        dummy_data.append({
            "image_path": image_path,
            "is_anomaly": True,
            "anomaly_type": "金属表面異常",
            "anomaly_description": f"金属プレート表面に微細な傷や変色が確認されました。製造工程での品質管理に注意が必要です。表面の仕上がりが基準値を下回っており、検査強化が推奨されます。",
            "confidence_level": random.randint(3, 5)
        })
    
    # チューブ画像用のダミーデータ  
    for i, image_path in enumerate(selected_tubes):
        dummy_data.append({
            "image_path": image_path,
            "is_anomaly": True,
            "anomaly_type": "形状異常",
            "anomaly_description": f"チューブの形状に軽微な歪みが見られます。成形工程での温度管理または圧力設定に問題がある可能性があります。品質基準の範囲内ですが、継続的な監視が必要です。",
            "confidence_level": random.randint(3, 5)
        })
    
    # JSONLファイルの作成
    import json
    dummy_jsonl_path = "data/text_generation/dummy_training_data.jsonl"
    dummy_dir = Path(dummy_jsonl_path).parent
    dummy_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dummy_jsonl_path, 'w', encoding='utf-8') as f:
        for data in dummy_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logger.info(f"ダミーデータ作成完了: {len(dummy_data)}サンプル")
    logger.info(f"JSONLファイル: {dummy_jsonl_path}")
    logger.info(f"使用した画像: {len(selected_metal)}枚の金属プレート + {len(selected_tubes)}枚のチューブ")
    
    return dummy_jsonl_path

def main():
    parser = argparse.ArgumentParser(description='MAD-FH LoRAテキスト生成モデル学習')
    parser.add_argument('--config', default='config/config.yaml', help='設定ファイルパス')
    parser.add_argument('--demo', action='store_true', help='デモモード（軽量学習）')
    parser.add_argument('--force-dummy', action='store_true', help='ダミーデータを強制使用')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== MAD-FH LoRAテキスト生成モデル学習開始 ===")
    
    try:
        # 環境チェック
        check_environment()
        
        # 設定読み込み
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"設定ファイル読み込み: {args.config}")
        
        # デモモード設定
        if args.demo:
            logger.info("デモモード: 軽量学習設定を適用")
            config['text_generation']['training']['epochs'] = 2
            config['text_generation']['training']['batch_size'] = 1
            config['text_generation']['training']['warmup_steps'] = 10
        
        # フィードバックデータの確認
        from src.ui.feedback_manager import FeedbackDataManager
        
        feedback_db_path = config['text_generation']['data']['feedback_db_path']
        feedback_manager = FeedbackDataManager(feedback_db_path)
        stats = feedback_manager.get_feedback_statistics()
        
        anomaly_count = stats['anomaly_normal_ratio'].get(1, 0)
        logger.info(f"フィードバックDB異常データ数: {anomaly_count}")
        
        # データの準備
        if args.force_dummy or anomaly_count < 3:
            logger.info("ダミーデータを使用します")
            training_data_path = prepare_dummy_data(config)
        else:
            logger.info("フィードバックデータから学習用データセットを作成")
            output_dir = "data/text_generation"
            dataset_info = feedback_manager.create_training_dataset(output_dir)
            training_data_path = dataset_info['jsonl_path']
            logger.info(f"学習用データセット: {dataset_info['total_samples']}サンプル")
        
        # LoRAトレーナーの初期化と学習実行
        from src.lora.lora_trainer import LoRAMultimodalTrainer
        
        logger.info("LoRAトレーナー初期化中...")
        trainer = LoRAMultimodalTrainer(args.config)
        
        # 学習設定の上書き（デモモード等）
        if args.demo:
            trainer.config = config
        
        # データ準備
        logger.info("学習データ準備中...")
        if args.force_dummy or anomaly_count < 3:
            # ダミーデータのパスを渡す
            trainer.prepare_data(dummy_jsonl_path=str(training_data_path))
        else:
            trainer.prepare_data()
        
        # 学習実行
        logger.info("LoRA学習開始...")
        train_result = trainer.train()
        
        # 結果表示
        logger.info("=== 学習完了 ===")
        logger.info(f"学習時間: {train_result.metrics.get('train_runtime', 0):.2f}秒")
        logger.info(f"最終損失: {train_result.metrics.get('train_loss', 'N/A')}")
        
        # 評価実行
        logger.info("モデル評価中...")
        eval_results = trainer.evaluate()
        logger.info(f"評価結果: {eval_results}")
        
        # 推論テスト
        logger.info("推論テスト実行中...")
        from src.lora.inference import AnomalyDescriptionInference
        
        inference_engine = AnomalyDescriptionInference(args.config)
        inference_engine.load_model()
        
        # テスト画像での推論
        from PIL import Image
        test_image = Image.new('RGB', (512, 512), color=(200, 100, 100))
        
        result = inference_engine.predict_single(
            test_image,
            prompt="この画像の異常を説明してください:",
            max_new_tokens=64
        )
        
        logger.info("=== 推論テスト結果 ===")
        if result.get('success', True):
            logger.info(f"生成説明: {result['description']}")
            logger.info(f"確信度: {result.get('confidence', 'N/A')}")
        else:
            logger.error(f"推論エラー: {result.get('error', '不明')}")
        
        logger.info("=== LoRAテキスト生成モデル学習完了 ===")
        logger.info(f"モデル保存先: {config['text_generation']['model_path']}")
        logger.info("Streamlit UIの 'AI説明生成' タブで使用できます")
        
    except Exception as e:
        logger.error(f"学習中にエラーが発生: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # GPUメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPUメモリをクリアしました")

if __name__ == "__main__":
    main()
