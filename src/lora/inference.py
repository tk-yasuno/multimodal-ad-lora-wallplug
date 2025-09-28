"""
LoRAマルチモーダルモデルの推論・デプロイクラス
"""

import os
import torch
import yaml
import json
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from datetime import datetime

from .multimodal_model import LoRAMultimodalModel

logger = logging.getLogger(__name__)

class AnomalyDescriptionInference:
    """
    異常説明生成の推論クラス
    """
    
    def __init__(self, config_path: str = "config/config.yaml", model_path: str = None):
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # モデルパスの設定
        if model_path:
            self.config['text_generation']['model_path'] = model_path
        
        # モデルの初期化
        self.model = LoRAMultimodalModel(self.config)
        self.is_loaded = False
        
        # 推論履歴
        self.inference_history = []
        
        logger.info("異常説明推論クラス初期化完了")
    
    def load_model(self, force_reload: bool = False):
        """モデルの読み込み"""
        if self.is_loaded and not force_reload:
            return
        
        try:
            model_path = Path(self.config['text_generation']['model_path'])
            
            if model_path.exists():
                logger.info(f"学習済みLoRAモデル読み込み: {model_path}")
                self.model.load_model(load_pretrained=True)
            else:
                logger.info("ベースモデルのみ読み込み（LoRA未学習）")
                self.model.load_model(load_pretrained=False)
            
            self.is_loaded = True
            logger.info("モデル読み込み完了")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def predict_single(
        self,
        image: Union[Image.Image, str, np.ndarray],
        prompt: str = None,
        include_confidence: bool = True,
        **generation_kwargs
    ) -> Dict[str, Union[str, float]]:
        """
        単一画像の異常説明生成
        
        Args:
            image: 入力画像（PIL Image、ファイルパス、またはnumpy配列）
            prompt: カスタムプロンプト
            include_confidence: 確信度推定を含めるか
            **generation_kwargs: 生成パラメータ
        
        Returns:
            生成結果の辞書
        """
        if not self.is_loaded:
            self.load_model()
        
        # 画像の前処理
        processed_image = self._preprocess_image(image)
        
        # プロンプトの設定
        if prompt is None:
            prompt = "この画像の異常を詳しく説明してください:"
        
        try:
            # 異常説明の生成
            description = self.model.generate_description(
                processed_image,
                prompt=prompt,
                **generation_kwargs
            )
            
            # 結果の構成
            result = {
                'description': description,
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            # 確信度の推定（簡易実装）
            if include_confidence:
                confidence = self._estimate_confidence(description)
                result['confidence'] = confidence
            
            # 履歴に記録
            self.inference_history.append(result.copy())
            
            logger.info(f"推論完了: {description[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return {
                'description': "異常説明の生成に失敗しました",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(
        self,
        images: List[Union[Image.Image, str, np.ndarray]],
        prompts: List[str] = None,
        **generation_kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        """
        バッチ推論
        
        Args:
            images: 入力画像のリスト
            prompts: プロンプトのリスト
            **generation_kwargs: 生成パラメータ
        
        Returns:
            生成結果のリスト
        """
        if not self.is_loaded:
            self.load_model()
        
        if prompts is None:
            prompts = ["この画像の異常を詳しく説明してください:"] * len(images)
        
        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            logger.info(f"バッチ推論進行中: {i+1}/{len(images)}")
            result = self.predict_single(
                image, prompt, include_confidence=True, **generation_kwargs
            )
            results.append(result)
        
        return results
    
    def _preprocess_image(self, image: Union[Image.Image, str, np.ndarray]) -> Image.Image:
        """画像の前処理"""
        if isinstance(image, str):
            # ファイルパスの場合
            if os.path.exists(image):
                return Image.open(image).convert('RGB')
            else:
                # Base64エンコードされた画像の場合
                try:
                    image_data = base64.b64decode(image)
                    return Image.open(BytesIO(image_data)).convert('RGB')
                except Exception:
                    raise ValueError(f"無効な画像パス/データ: {image}")
        
        elif isinstance(image, np.ndarray):
            # numpy配列の場合
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image).convert('RGB')
        
        elif isinstance(image, Image.Image):
            # PILイメージの場合
            return image.convert('RGB')
        
        else:
            raise ValueError(f"サポートされていない画像形式: {type(image)}")
    
    def _estimate_confidence(self, description: str) -> float:
        """
        生成された説明から確信度を推定（簡易実装）
        
        Args:
            description: 生成された説明文
        
        Returns:
            確信度（0.0-1.0）
        """
        # 説明文の特徴から確信度を推定
        confidence_indicators = {
            '明らかに': 0.9,
            'はっきりと': 0.85,
            '確実に': 0.9,
            '疑いなく': 0.95,
            'おそらく': 0.6,
            'と思われます': 0.5,
            'かもしれません': 0.4,
            '不明': 0.2,
            '判断できません': 0.1
        }
        
        # 基本確信度
        base_confidence = 0.5
        
        # キーワードベースの調整
        description_lower = description.lower()
        for keyword, confidence_boost in confidence_indicators.items():
            if keyword in description_lower:
                base_confidence = max(base_confidence, confidence_boost)
        
        # 説明文の長さによる調整
        description_length = len(description)
        if description_length > 50:
            base_confidence += 0.1  # 詳細な説明は高い確信度
        elif description_length < 20:
            base_confidence -= 0.1  # 短い説明は低い確信度
        
        # 0.0-1.0の範囲にクリップ
        return max(0.0, min(1.0, base_confidence))
    
    def get_inference_statistics(self) -> Dict[str, Union[int, float, List]]:
        """推論統計の取得"""
        if not self.inference_history:
            return {'total_inferences': 0}
        
        # 基本統計
        total_inferences = len(self.inference_history)
        
        # 確信度統計
        confidences = [
            result.get('confidence', 0.5) 
            for result in self.inference_history 
            if 'confidence' in result
        ]
        
        stats = {
            'total_inferences': total_inferences,
            'average_confidence': np.mean(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0,
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.4),
        }
        
        # 最近の推論（最新10件）
        recent_inferences = self.inference_history[-10:]
        stats['recent_inferences'] = [
            {
                'description': result['description'][:100] + '...' if len(result['description']) > 100 else result['description'],
                'confidence': result.get('confidence', 'N/A'),
                'timestamp': result['timestamp']
            }
            for result in recent_inferences
        ]
        
        return stats
    
    def save_inference_history(self, output_path: str = None):
        """推論履歴の保存"""
        if output_path is None:
            output_path = f"data/text_generation/inference_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'statistics': self.get_inference_statistics(),
                    'history': self.inference_history
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"推論履歴保存完了: {output_path}")
            
        except Exception as e:
            logger.error(f"推論履歴保存エラー: {e}")

class AnomalyDescriptionAPI:
    """
    異常説明生成のAPI風ラッパークラス
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.inference_engine = AnomalyDescriptionInference(config_path)
        
    def initialize(self):
        """API初期化"""
        self.inference_engine.load_model()
        logger.info("異常説明生成API初期化完了")
    
    def generate_description(
        self,
        image_data: Union[str, bytes, Image.Image],
        prompt: str = None,
        options: Dict = None
    ) -> Dict[str, Union[str, float]]:
        """
        API形式での異常説明生成
        
        Args:
            image_data: 画像データ
            prompt: カスタムプロンプト
            options: 生成オプション
        
        Returns:
            生成結果
        """
        options = options or {}
        
        try:
            result = self.inference_engine.predict_single(
                image_data,
                prompt=prompt,
                **options
            )
            
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"API生成エラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Union[str, bool]]:
        """モデル情報の取得"""
        return {
            'model_name': self.inference_engine.config['text_generation']['model_name'],
            'model_loaded': self.inference_engine.is_loaded,
            'lora_enabled': True,
            'version': '1.0.0'
        }

# デバッグ・テスト用関数
def test_inference():
    """推論クラスのテスト"""
    import yaml
    
    # 設定読み込み
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 推論エンジンの初期化
    inference = AnomalyDescriptionInference()
    inference.load_model()
    
    # ダミー画像でテスト
    dummy_image = Image.new('RGB', (512, 512), color='red')
    
    # 推論実行
    result = inference.predict_single(
        dummy_image,
        prompt="この画像の異常を説明してください:"
    )
    
    print("=== 推論結果 ===")
    print(f"説明: {result['description']}")
    print(f"確信度: {result.get('confidence', 'N/A')}")
    
    # 統計情報
    stats = inference.get_inference_statistics()
    print(f"\n=== 推論統計 ===")
    print(f"総推論数: {stats['total_inferences']}")
    print(f"平均確信度: {stats['average_confidence']:.3f}")

if __name__ == "__main__":
    test_inference()
