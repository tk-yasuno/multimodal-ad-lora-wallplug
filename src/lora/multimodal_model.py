"""
LoRA対応マルチモーダル異常説明生成モデル
BLIP-2 + LoRAを使用した画像キャプション生成
"""

import os
import torch
import torch.nn as nn
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq,
    TrainingArguments, Trainer
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
import yaml
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from PIL import Image

logger = logging.getLogger(__name__)

class LoRAMultimodalModel:
    """
    LoRA対応のマルチモーダル異常説明生成モデル
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデル設定
        self.model_name = config['text_generation']['model_name']
        self.model_path = Path(config['text_generation']['model_path'])
        
        # LoRA設定
        self.lora_config = self._create_lora_config()
        
        # モデルとプロセッサの初期化
        self.processor = None
        self.model = None
        self.tokenizer = None
        
        logger.info(f"LoRAモデル初期化: {self.model_name}")
        logger.info(f"デバイス: {self.device}")
    
    def _create_lora_config(self) -> LoraConfig:
        """LoRA設定の作成"""
        lora_settings = self.config['text_generation']['lora']
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Vision-Language Modelingが無い場合はCAUSAL_LMを使用
            r=lora_settings['r'],
            lora_alpha=lora_settings['alpha'],
            lora_dropout=lora_settings['dropout'],
            target_modules=lora_settings['target_modules'],
            bias="none"
        )
    
    def load_model(self, load_pretrained: bool = True):
        """モデルとプロセッサの読み込み"""
        try:
            if load_pretrained and self.model_path.exists():
                # 学習済みLoRAモデルの読み込み
                logger.info(f"学習済みモデル読み込み: {self.model_path}")
                self._load_trained_model()
            else:
                # ベースモデルの読み込み
                logger.info(f"ベースモデル読み込み: {self.model_name}")
                self._load_base_model()
            
            # モデルをデバイスに移動
            self.model = self.model.to(self.device)
            
            logger.info("モデル読み込み完了")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def _load_base_model(self):
        """ベースモデルの読み込み"""
        logger.info(f"ベースモデル読み込み: {self.model_name}")
        
        # プロセッサとモデルの読み込み
        if "blip" in self.model_name.lower():
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            # BLIPモデルはdevice_map='auto'をサポートしていないため、手動でデバイスに移動
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            # モデルをCUDAデバイスに移動
            if torch.cuda.is_available() and self.device.type == 'cuda':
                self.model = self.model.to(self.device)
        else:
            # 他のVision-Languageモデル用の汎用的な読み込み
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        self.tokenizer = self.processor.tokenizer
        
        # 量子化の準備（メモリ効率化）
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRAの適用
        self.model = get_peft_model(self.model, self.lora_config)
        
        # 学習可能パラメータの表示
        self.model.print_trainable_parameters()
    
    def _load_trained_model(self):
        """学習済みLoRAモデルの読み込み"""
        from peft import PeftModel
        
        # ベースモデルを読み込み
        if "blip" in self.model_name.lower():
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            base_model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            base_model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        self.tokenizer = self.processor.tokenizer
        
        # LoRAアダプタを読み込み
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
    
    def generate_description(
        self,
        image: Image.Image,
        prompt: str = "この画像の異常を説明してください:",
        max_new_tokens: int = None,
        **kwargs
    ) -> str:
        """
        画像から異常説明を生成
        
        Args:
            image: 入力画像
            prompt: 生成プロンプト
            max_new_tokens: 生成する最大トークン数
            **kwargs: 生成パラメータ
        
        Returns:
            生成された異常説明テキスト
        """
        if self.model is None:
            raise ValueError("モデルが読み込まれていません")
        
        # 生成設定
        gen_config = self.config['text_generation']['generation']
        max_new_tokens = max_new_tokens or gen_config['max_new_tokens']
        
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': gen_config.get('temperature', 0.7),
            'top_p': gen_config.get('top_p', 0.9),
            'do_sample': gen_config.get('do_sample', True),
            'pad_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        
        try:
            # 入力の準備
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # 推論モードで生成
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # デコード
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # プロンプト部分を除去
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"テキスト生成エラー: {e}")
            return "異常説明の生成に失敗しました。"
    
    def batch_generate(
        self,
        images: List[Image.Image],
        prompts: List[str] = None,
        **kwargs
    ) -> List[str]:
        """
        バッチでの異常説明生成
        
        Args:
            images: 入力画像のリスト
            prompts: プロンプトのリスト
            **kwargs: 生成パラメータ
        
        Returns:
            生成された説明テキストのリスト
        """
        if prompts is None:
            prompts = ["この画像の異常を説明してください:"] * len(images)
        
        descriptions = []
        for image, prompt in zip(images, prompts):
            try:
                desc = self.generate_description(image, prompt, **kwargs)
                descriptions.append(desc)
            except Exception as e:
                logger.error(f"バッチ生成エラー: {e}")
                descriptions.append("生成エラー")
        
        return descriptions
    
    def save_model(self, output_dir: str = None):
        """LoRAモデルの保存"""
        if self.model is None:
            raise ValueError("保存するモデルがありません")
        
        output_dir = output_dir or str(self.model_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # LoRAアダプタのみを保存
            self.model.save_pretrained(str(output_path))
            
            # プロセッサも保存
            self.processor.save_pretrained(str(output_path))
            
            # 設定ファイルも保存
            config_path = output_path / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config['text_generation'], f, ensure_ascii=False, indent=2)
            
            logger.info(f"LoRAモデル保存完了: {output_path}")
            
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise

class AnomalyDescriptionEvaluator:
    """
    異常説明生成の評価クラス
    """
    
    def __init__(self, model: LoRAMultimodalModel):
        self.model = model
    
    def evaluate_on_test_set(
        self,
        test_images: List[Image.Image],
        ground_truth_descriptions: List[str],
        prompts: List[str] = None
    ) -> Dict[str, float]:
        """
        テストセットでの評価
        
        Args:
            test_images: テスト画像
            ground_truth_descriptions: 正解説明
            prompts: プロンプト
        
        Returns:
            評価メトリクス
        """
        generated_descriptions = self.model.batch_generate(
            test_images, prompts
        )
        
        # 簡易的な評価メトリクス
        metrics = {}
        
        # 長さの一致度
        length_similarity = self._calculate_length_similarity(
            generated_descriptions, ground_truth_descriptions
        )
        metrics['length_similarity'] = length_similarity
        
        # キーワード一致度
        keyword_overlap = self._calculate_keyword_overlap(
            generated_descriptions, ground_truth_descriptions
        )
        metrics['keyword_overlap'] = keyword_overlap
        
        return metrics
    
    def _calculate_length_similarity(
        self,
        generated: List[str],
        ground_truth: List[str]
    ) -> float:
        """テキスト長の類似度を計算"""
        similarities = []
        for gen, gt in zip(generated, ground_truth):
            len_gen, len_gt = len(gen), len(gt)
            similarity = 1 - abs(len_gen - len_gt) / max(len_gen, len_gt, 1)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_keyword_overlap(
        self,
        generated: List[str],
        ground_truth: List[str]
    ) -> float:
        """キーワード重複度を計算"""
        overlaps = []
        for gen, gt in zip(generated, ground_truth):
            gen_words = set(gen.split())
            gt_words = set(gt.split())
            
            if len(gt_words) == 0:
                overlap = 0.0
            else:
                overlap = len(gen_words & gt_words) / len(gt_words)
            
            overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

# デバッグ用の関数
def test_lora_model():
    """LoRAモデルのテスト"""
    import yaml
    
    # 設定読み込み
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # モデルの初期化
    lora_model = LoRAMultimodalModel(config)
    lora_model.load_model(load_pretrained=False)
    
    # ダミー画像でテスト
    dummy_image = Image.new('RGB', (512, 512), color='red')
    
    # 説明生成
    description = lora_model.generate_description(
        dummy_image,
        prompt="この画像の異常を説明してください:"
    )
    
    print(f"生成された説明: {description}")

if __name__ == "__main__":
    test_lora_model()
