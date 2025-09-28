"""
LoRA学習簡単テストスクリプト
基本的なLoRA学習機能の動作確認
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_lora_basic():
    """LoRA基本機能テスト"""
    print("🧪 LoRA基本機能テスト")
    print("="*40)
    
    try:
        # BLIPプロセッサーの基本テスト
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model_name = "Salesforce/blip2-opt-2.7b"
        print(f"📱 モデル読み込みテスト: {model_name}")
        
        # プロセッサーのみ読み込み
        processor = BlipProcessor.from_pretrained(model_name)
        print("✅ プロセッサー読み込み成功")
        
        # テスト画像準備
        data_dir = Path("data/processed/wallplugs/train/normal")
        if data_dir.exists():
            test_image_path = next(data_dir.glob("*.png"))
            test_image = Image.open(test_image_path).convert('RGB')
            print(f"✅ テスト画像読み込み: {test_image_path.name}")
            
            # プロセッサーテスト
            test_text = "Describe this wallplug:"
            inputs = processor(images=test_image, text=test_text, return_tensors="pt")
            print("✅ プロセッサー処理成功")
            print(f"   画像テンソル形状: {inputs['pixel_values'].shape}")
            print(f"   テキストトークン数: {inputs['input_ids'].shape[1]}")
            
        else:
            print("⚠️ テスト画像が見つかりません")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA基本テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lora_dataset():
    """LoRAデータセットテスト"""
    print("\n📊 LoRAデータセットテスト")
    print("="*40)
    
    try:
        from transformers import BlipProcessor
        
        # プロセッサー準備
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # データセットクラス準備（簡易版）
        data_dir = Path("data/processed/wallplugs")
        
        # サンプル取得
        train_normal_images = list((data_dir / "train" / "normal").glob("*.png"))[:3]
        train_anomal_images = list((data_dir / "train" / "anomalous").glob("*.png"))[:3]
        
        print(f"✅ 正常画像サンプル: {len(train_normal_images)}枚")
        print(f"✅ 異常画像サンプル: {len(train_anomal_images)}枚")
        
        # データ処理テスト
        for i, img_path in enumerate(train_normal_images):
            image = Image.open(img_path).convert('RGB')
            text = "This wallplug appears to be in normal condition."
            
            encoding = processor(
                images=image,
                text=text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            
            print(f"   サンプル{i+1}: {img_path.name} -> {encoding['pixel_values'].shape}")
        
        print("✅ データセット処理テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ データセットテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft_integration():
    """PEFT統合テスト"""
    print("\n🔧 PEFT統合テスト")
    print("="*40)
    
    try:
        from peft import LoraConfig, TaskType
        
        # LoRA設定テスト
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['q_proj', 'v_proj'],
            bias="none"
        )
        
        print("✅ LoRA設定作成成功")
        print(f"   Rank: {lora_config.r}")
        print(f"   Alpha: {lora_config.lora_alpha}")
        print(f"   Target modules: {lora_config.target_modules}")
        
        return True
        
    except Exception as e:
        print(f"❌ PEFT統合テストエラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 LoRA学習事前テスト")
    print("="*50)
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"   メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️ GPU not available")
    
    # 各テスト実行
    tests_passed = 0
    
    if test_lora_basic():
        tests_passed += 1
    
    if test_lora_dataset():
        tests_passed += 1
    
    if test_peft_integration():
        tests_passed += 1
    
    # 結果表示
    print(f"\n📊 テスト結果: {tests_passed}/3 passed")
    
    if tests_passed == 3:
        print("✅ すべてのテストが成功！LoRA学習準備完了")
        print("\n💡 次のステップ:")
        print("   1. train_lora_wallplugs.py で実際の学習実行")
        print("   2. エポック数・バッチサイズを環境に応じて調整")
        return True
    else:
        print("❌ 一部のテストが失敗しました")
        return False

if __name__ == "__main__":
    main()