"""
MVTec AD Wallplugs × MiniCPM + LoRA 軽量学習スクリプト
リソース制限環境での学習実行
"""

import sys
import torch
from pathlib import Path
import json

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_system_resources():
    """システムリソース確認"""
    print("🔍 システムリソース確認")
    print("="*40)
    
    # GPU確認
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"   メモリ: {gpu_memory:.1f}GB")
        
        # VRAM使用量確認
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   使用中: {allocated:.2f}GB")
        print(f"   予約済み: {reserved:.2f}GB")
        
        return True, gpu_memory
    else:
        print("⚠️  GPU not available, using CPU")
        return False, 0
    
def quick_model_test():
    """軽量モデルテスト"""
    print("\n🧪 軽量モデルテスト")
    print("="*40)
    
    try:
        # MiniCPM基本テスト
        from src.models.minicpm_autoencoder import MiniCPMVisionEncoder
        print("✅ MiniCPM モジュールインポート成功")
        
        # 軽量設定でのモデル初期化テスト
        test_config = {
            'input_channels': 3,
            'latent_dim': 256,  # 軽量化
            'input_size': (512, 512),  # サイズ削減
            'use_minicpm': False,  # 初回はMiniCPMを無効化
            'anomaly_threshold': 0.1
        }
        
        from src.models.minicpm_autoencoder import MiniCPMAnomalyDetector
        detector = MiniCPMAnomalyDetector(test_config)
        print("✅ 軽量異常検知モデル初期化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ モデルテストエラー: {e}")
        return False

def run_lightweight_training():
    """軽量学習実行"""
    print("\n🚀 軽量学習開始")
    print("="*40)
    
    # リソース確認
    has_gpu, gpu_memory = check_system_resources()
    
    # モデルテスト
    if not quick_model_test():
        print("❌ モデルテストに失敗しました")
        return False
    
    # 学習設定調整
    if gpu_memory < 8.0:
        print("⚠️  GPU メモリ不足のため軽量設定を使用")
        batch_size = 1
        latent_dim = 256
        use_minicpm = False  # メモリ節約のため無効化
        epochs = 5
    else:
        print("✅ 通常設定で学習実行")
        batch_size = 2
        latent_dim = 512
        use_minicpm = True
        epochs = 10
    
    print(f"\n📋 学習設定:")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   潜在次元: {latent_dim}")
    print(f"   MiniCPM使用: {use_minicpm}")
    print(f"   エポック数: {epochs}")
    
    # 軽量学習設定ファイル作成
    config = {
        'minicpm_anomaly': {
            'model': {
                'latent_dim': latent_dim,
                'use_minicpm': use_minicpm,
                'minicpm_weight': 0.3,
                'anomaly_threshold': 0.1
            },
            'training': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'patience': 5
            }
        },
        'lora_explanation': {
            'model': {
                'name': 'Salesforce/blip2-opt-2.7b'
            },
            'lora': {
                'r': 8,  # 軽量化
                'alpha': 16,
                'dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj']
            },
            'training': {
                'epochs': 5,  # 軽量化
                'batch_size': 1,
                'learning_rate': 5e-5,
                'weight_decay': 0.01,
                'warmup_steps': 20
            }
        }
    }
    
    # 設定保存
    config_path = Path("lightweight_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 軽量設定保存: {config_path}")
    
    return True

def main():
    """メイン実行"""
    print("🚀 MVTec AD Wallplugs 軽量学習準備")
    print("="*50)
    
    # データ確認
    data_dir = Path("data/processed/wallplugs")
    if not data_dir.exists():
        print("❌ 前処理済みデータが見つかりません")
        print("   preprocess_mvtec.py を先に実行してください")
        return False
    
    # リソース・モデル確認
    if not run_lightweight_training():
        return False
    
    print("\n✅ 軽量学習準備完了！")
    print("\n💡 次のステップ:")
    print("1. 異常検知モデル単体テスト:")
    print("   python train_minicpm_wallplugs.py")
    print("2. LoRAモデル単体テスト:")  
    print("   python train_lora_wallplugs.py")
    print("3. 統合学習実行:")
    print("   python train_wallplugs_integrated.py")
    
    return True

if __name__ == "__main__":
    main()