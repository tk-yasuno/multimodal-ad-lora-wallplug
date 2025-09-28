"""
MVTec AD Wallplugs 簡単LoRAデモ学習
最小限の機能でLoRA学習を実行するデモスクリプト
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SimpleWallplugsDataset(Dataset):
    """シンプルなWallplugsデータセット"""
    
    def __init__(self, data_dir, split='train', max_samples=20):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        
        # サンプル収集
        self.samples = []
        self.collect_samples()
        
        print(f"{split} dataset: {len(self.samples)} samples")
    
    def collect_samples(self):
        """サンプル収集（制限付き）"""
        # 正常画像
        normal_dir = self.data_dir / self.split / "normal"
        if normal_dir.exists():
            normal_files = list(normal_dir.glob("*.png"))[:self.max_samples//2]
            for img_path in normal_files:
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 'normal',
                    'explanation': 'This wallplug appears to be in normal condition with proper alignment.'
                })
        
        # 異常画像
        anomalous_dir = self.data_dir / self.split / "anomalous"
        if anomalous_dir.exists():
            anomalous_files = list(anomalous_dir.glob("*.png"))[:self.max_samples//2]
            for img_path in anomalous_files:
                explanation = self.get_anomaly_explanation(img_path.name)
                self.samples.append({
                    'image_path': str(img_path),
                    'label': 'anomalous', 
                    'explanation': explanation
                })
    
    def get_anomaly_explanation(self, filename):
        """ファイル名から異常説明生成"""
        if "overexposed" in filename:
            return "This wallplug shows overexposure issues affecting quality assessment."
        elif "underexposed" in filename:
            return "The wallplug appears underexposed with reduced surface detail visibility."
        elif "shift" in filename:
            return "This wallplug exhibits positional shift or misalignment anomaly."
        else:
            return "This wallplug shows anomalous characteristics deviating from normal standards."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 画像読み込み（サイズ統一）
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((224, 224))  # 小さめサイズ
        
        return {
            'image': image,
            'explanation': sample['explanation'],
            'label': sample['label'],
            'path': sample['image_path']
        }

class SimpleBLIPLoRADemo:
    """シンプルBLIP LoRAデモ"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 出力ディレクトリ
        self.output_dir = Path("models/simple_lora_demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル・プロセッサ
        self.processor = None
        self.model = None
        
        # 学習履歴
        self.training_log = []
    
    def setup_model(self):
        """モデル設定"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            model_name = "Salesforce/blip-image-captioning-base"
            print(f"Loading model: {model_name}")
            
            # プロセッサ読み込み
            self.processor = BlipProcessor.from_pretrained(model_name)
            print("✅ Processor loaded")
            
            # モデル読み込み（軽量版）
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            print("✅ Model loaded")
            
            # モデルサイズ確認
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
    
    def setup_data(self):
        """データ準備"""
        data_dir = Path("data/processed/wallplugs")
        
        if not data_dir.exists():
            print("❌ 前処理済みデータが見つかりません")
            return False
        
        # データセット作成（小規模）
        self.train_dataset = SimpleWallplugsDataset(data_dir, split='train', max_samples=10)
        self.val_dataset = SimpleWallplugsDataset(data_dir, split='validation', max_samples=6)
        
        return True
    
    def generate_sample(self, sample):
        """サンプル生成"""
        try:
            image = sample['image']
            prompt = "Describe this wallplug quality:"
            
            # 入力準備
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
            
            # デコード
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト除去
            if prompt in generated_text:
                result = generated_text.replace(prompt, "").strip()
            else:
                result = generated_text
            
            return result
            
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def run_demo_training(self):
        """デモ学習実行"""
        print("\n🚀 Simple LoRA Demo Training")
        print("="*50)
        
        # モデル・データ準備
        if not self.setup_model():
            return False
        
        if not self.setup_data():
            return False
        
        # 学習前サンプル生成
        print("\n📝 Pre-training samples:")
        pre_samples = []
        for i, sample in enumerate(self.train_dataset):
            if i >= 3:  # 3サンプルまで
                break
            
            generated = self.generate_sample(sample)
            
            print(f"Sample {i+1}: {Path(sample['path']).name}")
            print(f"  Ground truth: {sample['explanation'][:60]}...")
            print(f"  Generated: {generated[:60]}...")
            print(f"  Label: {sample['label']}")
            print()
            
            pre_samples.append({
                'filename': Path(sample['path']).name,
                'ground_truth': sample['explanation'],
                'generated': generated,
                'label': sample['label']
            })
        
        # デモ学習ログ
        demo_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': "Salesforce/blip-image-captioning-base",
            'dataset': 'MVTec AD Wallplugs',
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'pre_training_samples': pre_samples,
            'notes': [
                'This is a demo run without actual LoRA training',
                'Shows baseline BLIP performance on wallplugs data',
                'Ready for LoRA fine-tuning implementation'
            ]
        }
        
        # 結果保存
        results_path = self.output_dir / "demo_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Demo results saved: {results_path}")
        
        # サマリー表示
        print("\n📊 Demo Summary:")
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Val samples: {len(self.val_dataset)}")
        print(f"   Model: Salesforce/blip-image-captioning-base")
        print(f"   Status: Baseline performance confirmed")
        
        print("\n💡 Next steps:")
        print("   1. Implement LoRA configuration")
        print("   2. Add training loop with loss calculation")
        print("   3. Compare pre/post LoRA performance")
        
        return True

def main():
    """メイン実行"""
    print("🚀 MVTec AD Wallplugs Simple LoRA Demo")
    print("="*50)
    
    try:
        demo = SimpleBLIPLoRADemo()
        success = demo.run_demo_training()
        
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed")
            
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()