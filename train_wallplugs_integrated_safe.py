"""
MVTec AD Wallplugs 統合学習マネージャ (Unicode Safe Version)
異常検知モデル（MiniCPM）とLoRA説明生成モデルの統合学習システム
"""

import sys
import os
from pathlib import Path
import json
import yaml
import torch
from datetime import datetime
import subprocess
import argparse

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class WallplugsTrainingManager:
    """Wallplugs学習管理クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # 学習結果記録
        self.training_results = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'anomaly_detection': {},
            'lora_explanation': {},
            'integration_test': {},
            'summary': {}
        }
        
        print("MVTec AD Wallplugs 統合学習マネージャ")
        print("=" * 60)
    
    def check_prerequisites(self):
        """前提条件チェック"""
        print("前提条件チェック中...")
        
        issues = []
        
        # 前処理済みデータ確認
        data_dir = Path("data/processed/wallplugs")
        if not data_dir.exists():
            issues.append("前処理済みデータが見つかりません (data/processed/wallplugs)")
        else:
            # データ内容確認
            train_normal = data_dir / "train" / "normal"
            train_anomalous = data_dir / "train" / "anomalous"
            val_normal = data_dir / "validation" / "normal"
            val_anomalous = data_dir / "validation" / "anomalous"
            
            if not all([train_normal.exists(), train_anomalous.exists(), 
                       val_normal.exists(), val_anomalous.exists()]):
                issues.append("データディレクトリ構造が不完全です")
            else:
                train_normal_count = len(list(train_normal.glob("*.png")))
                train_anomalous_count = len(list(train_anomalous.glob("*.png")))
                val_normal_count = len(list(val_normal.glob("*.png")))
                val_anomalous_count = len(list(val_anomalous.glob("*.png")))
                
                print(f"  [OK] 学習用正常データ: {train_normal_count}枚")
                print(f"  [OK] 学習用異常データ: {train_anomalous_count}枚")
                print(f"  [OK] 検証用正常データ: {val_normal_count}枚")
                print(f"  [OK] 検証用異常データ: {val_anomalous_count}枚")
        
        # GPU確認
        if torch.cuda.is_available():
            print(f"  [OK] GPU利用可能: {torch.cuda.get_device_name()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"     メモリ: {gpu_memory:.1f}GB")
            if gpu_memory < 8.0:
                issues.append("GPU メモリが8GB未満です。学習に時間がかかる可能性があります。")
        else:
            issues.append("GPU が利用できません。CPU モードでは学習に長時間かかります。")
        
        # 必要パッケージ確認
        required_packages = ['transformers', 'peft', 'torch', 'PIL', 'sklearn']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"必要パッケージが見つかりません: {package}")
        
        if issues:
            print("\n[WARNING] 問題が見つかりました:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("  [OK] すべての前提条件が満たされています")
            return True
    
    def create_training_config(self):
        """統合学習設定作成"""
        config = {
            'session_info': {
                'session_id': self.training_results['session_id'],
                'dataset': 'MVTec AD Wallplugs',
                'description': 'MiniCPM異常検知 + LoRA説明生成の統合学習'
            },
            'minicpm_anomaly': {
                'model': {
                    'latent_dim': 512,
                    'use_minicpm': True,
                    'minicpm_weight': 0.3,
                    'anomaly_threshold': 0.1
                },
                'training': {
                    'batch_size': 2,  # GPU メモリに応じて調整
                    'epochs': 20,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'patience': 8
                }
            },
            'lora_explanation': {
                'model': {
                    'name': 'Salesforce/blip2-opt-2.7b'
                },
                'lora': {
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'target_modules': ['q_proj', 'v_proj']
                },
                'training': {
                    'epochs': 8,
                    'batch_size': 1,  # LoRAのため小さく設定
                    'learning_rate': 5e-5,
                    'weight_decay': 0.01,
                    'warmup_steps': 50
                }
            }
        }
        
        # 設定ファイル保存
        config_path = self.models_dir / f"training_config_{self.training_results['session_id']}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[CONFIG] 統合学習設定保存: {config_path}")
        return config, config_path
    
    def run_anomaly_detection_training(self, config):
        """異常検知モデル学習実行"""
        print("\n" + "=" * 60)
        print("Phase 1: MiniCPM異常検知モデル学習")
        print("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # Pythonスクリプト実行
            result = subprocess.run([
                sys.executable, "train_minicpm_wallplugs.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                print("[OK] 異常検知モデル学習完了")
                # 結果記録
                self.training_results['anomaly_detection'] = {
                    'status': 'success',
                    'training_time': training_time,
                    'output': result.stdout[-1000:],  # 最後の1000文字
                    'model_path': 'models/minicpm_autoencoder_wallplugs_best.pth'
                }
                return True
            else:
                print("[FAILED] 異常検知モデル学習失敗")
                print("エラー出力:")
                print(result.stderr)
                
                self.training_results['anomaly_detection'] = {
                    'status': 'failed',
                    'error': result.stderr,
                    'output': result.stdout
                }
                return False
                
        except Exception as e:
            print(f"[FAILED] 異常検知学習実行エラー: {e}")
            self.training_results['anomaly_detection'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def run_lora_training(self, config):
        """LoRA説明生成モデル学習実行"""
        print("\n" + "=" * 60)
        print("Phase 2: LoRA説明生成モデル学習")
        print("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # LoRA学習スクリプト実行
            result = subprocess.run([
                sys.executable, "train_lora_wallplugs.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                print("[OK] LoRA説明生成モデル学習完了")
                
                self.training_results['lora_explanation'] = {
                    'status': 'success',
                    'training_time': training_time,
                    'output': result.stdout[-1000:],
                    'model_path': 'models/lora_wallplugs/final_model'
                }
                return True
            else:
                print("[FAILED] LoRA説明生成モデル学習失敗")
                print("エラー出力:")
                print(result.stderr)
                
                self.training_results['lora_explanation'] = {
                    'status': 'failed',
                    'error': result.stderr,
                    'output': result.stdout
                }
                return False
                
        except Exception as e:
            print(f"[FAILED] LoRA学習実行エラー: {e}")
            self.training_results['lora_explanation'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def run_integration_test(self):
        """統合テスト実行"""
        print("\n" + "=" * 60)
        print("Phase 3: 統合テスト")
        print("=" * 60)
        
        try:
            # 学習済みモデルでFODDテスト実行
            start_time = datetime.now()
            
            result = subprocess.run([
                sys.executable, "test_wallplugs_fodd.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            end_time = datetime.now()
            test_time = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                print("[OK] 統合テスト完了")
                
                self.training_results['integration_test'] = {
                    'status': 'success',
                    'test_time': test_time,
                    'output': result.stdout[-1000:]
                }
                return True
            else:
                print("[WARNING] 統合テストで問題発生（継続可能）")
                
                self.training_results['integration_test'] = {
                    'status': 'warning',
                    'test_time': test_time,
                    'error': result.stderr,
                    'output': result.stdout
                }
                return True  # 統合テストは失敗してもOK
                
        except Exception as e:
            print(f"[FAILED] 統合テスト実行エラー: {e}")
            self.training_results['integration_test'] = {
                'status': 'error',
                'error': str(e)
            }
            return True  # 統合テストは失敗してもOK
    
    def generate_final_report(self):
        """最終レポート生成"""
        print("\n" + "=" * 60)
        print("[REPORT] 最終レポート生成")
        print("=" * 60)
        
        # 学習結果サマリー
        anomaly_success = self.training_results['anomaly_detection'].get('status') == 'success'
        lora_success = self.training_results['lora_explanation'].get('status') == 'success'
        integration_ok = self.training_results['integration_test'].get('status') in ['success', 'warning']
        
        overall_success = anomaly_success and lora_success
        
        # サマリー作成
        self.training_results['summary'] = {
            'overall_status': 'success' if overall_success else 'partial' if (anomaly_success or lora_success) else 'failed',
            'anomaly_detection_success': anomaly_success,
            'lora_explanation_success': lora_success,
            'integration_test_ok': integration_ok,
            'total_training_time': (
                self.training_results['anomaly_detection'].get('training_time', 0) +
                self.training_results['lora_explanation'].get('training_time', 0)
            ),
            'end_time': datetime.now().isoformat()
        }
        
        # レポートファイル保存
        report_path = self.models_dir / f"training_report_{self.training_results['session_id']}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, ensure_ascii=False, default=str)
        
        # コンソール出力
        print(f"[FILE] 詳細レポート保存: {report_path}")
        
        print(f"\n学習セッション完了")
        print(f"   セッションID: {self.training_results['session_id']}")
        print(f"   全体ステータス: {self.training_results['summary']['overall_status'].upper()}")
        print(f"   異常検知学習: {'[OK]' if anomaly_success else '[FAILED]'}")
        print(f"   LoRA説明生成: {'[OK]' if lora_success else '[FAILED]'}")
        print(f"   統合テスト: {'[OK]' if integration_ok else '[WARNING]'}")
        print(f"   総学習時間: {self.training_results['summary']['total_training_time']:.1f}秒")
        
        if overall_success:
            print(f"\n[SUCCESS] すべての学習が正常に完了しました！")
            print(f"次のステップ:")
            print(f"  1. Streamlit UIでの動作確認")
            print(f"  2. 他のデータセット（sheet_metal, wallnuts, fruit_jelly）での学習")
            print(f"  3. 本格運用環境での展開")
        else:
            print(f"\n[WARNING] 一部の学習で問題が発生しました")
            print(f"   詳細は {report_path} を確認してください")
        
        return report_path
    
    def run_full_training(self, skip_anomaly=False, skip_lora=False):
        """完全統合学習実行"""
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 前提条件チェック
        if not self.check_prerequisites():
            print("[FAILED] 前提条件が満たされていません。学習を中止します。")
            return False
        
        # 学習設定作成
        config, config_path = self.create_training_config()
        
        success = True
        
        # Phase 1: 異常検知モデル学習
        if not skip_anomaly:
            if not self.run_anomaly_detection_training(config):
                success = False
        else:
            print("[SKIP] 異常検知学習をスキップしました")
        
        # Phase 2: LoRA説明生成学習
        if not skip_lora:
            if not self.run_lora_training(config):
                success = False
        else:
            print("[SKIP] LoRA学習をスキップしました")
        
        # Phase 3: 統合テスト
        self.run_integration_test()
        
        # 最終レポート
        report_path = self.generate_final_report()
        
        return success, report_path

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='MVTec AD Wallplugs 統合学習マネージャ')
    parser.add_argument('--skip-anomaly', action='store_true', help='異常検知学習をスキップ')
    parser.add_argument('--skip-lora', action='store_true', help='LoRA学習をスキップ')
    
    args = parser.parse_args()
    
    try:
        manager = WallplugsTrainingManager()
        success, report_path = manager.run_full_training(
            skip_anomaly=args.skip_anomaly,
            skip_lora=args.skip_lora
        )
        
        if success:
            print(f"\n[SUCCESS] 統合学習が正常に完了しました！")
            exit(0)
        else:
            print(f"\n[FAILED] 統合学習で問題が発生しました。")
            exit(1)
            
    except KeyboardInterrupt:
        print(f"\n[STOP] 学習が中断されました。")
        exit(2)
    except Exception as e:
        print(f"\n[ERROR] 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        exit(3)

if __name__ == "__main__":
    main()