"""
LoRAテキスト生成機能をUIに統合するためのユーティリティクラス
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.lora.inference import AnomalyDescriptionInference, AnomalyDescriptionAPI

logger = logging.getLogger(__name__)

class UITextGenerationIntegration:
    """
    UIとテキスト生成機能の統合クラス
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.text_generator = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """テキスト生成機能の初期化"""
        try:
            logger.info("テキスト生成機能初期化中...")
            
            # APIクラスの初期化
            self.text_generator = AnomalyDescriptionAPI(self.config_path)
            self.text_generator.initialize()
            
            self.is_initialized = True
            logger.info("テキスト生成機能初期化完了")
            return True
            
        except Exception as e:
            logger.error(f"テキスト生成機能初期化エラー: {e}")
            self.is_initialized = False
            return False
    
    def generate_anomaly_description(
        self,
        image: Union[Image.Image, str],
        custom_prompt: str = None,
        generation_params: Dict = None
    ) -> Dict[str, Union[str, float, bool]]:
        """
        異常説明の生成
        
        Args:
            image: 入力画像
            custom_prompt: カスタムプロンプト
            generation_params: 生成パラメータ
        
        Returns:
            生成結果の辞書
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'テキスト生成機能が初期化されていません',
                'description': '',
                'confidence': 0.0
            }
        
        try:
            # デフォルトプロンプト
            if custom_prompt is None:
                custom_prompt = "この画像の異常を詳しく説明してください:"
            
            # デフォルト生成パラメータ
            default_params = {
                'max_new_tokens': 128,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            }
            
            if generation_params:
                default_params.update(generation_params)
            
            # 生成実行
            result = self.text_generator.generate_description(
                image_data=image,
                prompt=custom_prompt,
                options=default_params
            )
            
            if result['success']:
                return {
                    'success': True,
                    'description': result['result']['description'],
                    'confidence': result['result'].get('confidence', 0.5),
                    'timestamp': result['result']['timestamp']
                }
            else:
                return {
                    'success': False,
                    'error': result['error'],
                    'description': '',
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"異常説明生成エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'description': '',
                'confidence': 0.0
            }
    
    def get_generation_suggestions(
        self,
        anomaly_type: str = None
    ) -> List[str]:
        """
        異常タイプに基づく生成プロンプトの提案
        
        Args:
            anomaly_type: 異常タイプ
        
        Returns:
            プロンプト候補のリスト
        """
        base_prompts = [
            "この画像の異常を詳しく説明してください:",
            "画像に見られる問題点を分析してください:",
            "この画像の異常な部分を特定して説明してください:",
        ]
        
        if anomaly_type:
            specific_prompts = {
                '設備故障': [
                    "この設備の故障状況を詳しく説明してください:",
                    "機械の損傷や故障箇所を特定してください:",
                    "設備の異常な状態を技術的に説明してください:"
                ],
                '製品欠陥': [
                    "この製品の欠陥を詳しく説明してください:",
                    "製品の品質問題を分析してください:",
                    "製品の不良箇所を特定して説明してください:"
                ],
                '安全問題': [
                    "この画像の安全上の問題を説明してください:",
                    "安全規則違反や危険要素を特定してください:",
                    "作業安全に関する問題点を分析してください:"
                ],
                '環境異常': [
                    "環境の異常状態を詳しく説明してください:",
                    "環境条件の問題点を分析してください:",
                    "環境異常の影響を評価してください:"
                ]
            }
            
            if anomaly_type in specific_prompts:
                return specific_prompts[anomaly_type] + base_prompts
        
        return base_prompts
    
    def get_model_status(self) -> Dict[str, Union[str, bool]]:
        """モデルの状態情報を取得"""
        if not self.is_initialized:
            return {
                'status': 'not_initialized',
                'model_loaded': False,
                'error': 'テキスト生成機能が初期化されていません'
            }
        
        try:
            model_info = self.text_generator.get_model_info()
            return {
                'status': 'ready' if model_info['model_loaded'] else 'loading',
                'model_name': model_info['model_name'],
                'model_loaded': model_info['model_loaded'],
                'lora_enabled': model_info['lora_enabled'],
                'version': model_info['version']
            }
        except Exception as e:
            return {
                'status': 'error',
                'model_loaded': False,
                'error': str(e)
            }
    
    def batch_generate(
        self,
        images: List[Union[Image.Image, str]],
        prompts: List[str] = None
    ) -> List[Dict[str, Union[str, float, bool]]]:
        """
        バッチでの異常説明生成
        
        Args:
            images: 画像のリスト
            prompts: プロンプトのリスト
        
        Returns:
            生成結果のリスト
        """
        if not self.is_initialized:
            return [
                {
                    'success': False,
                    'error': 'テキスト生成機能が初期化されていません',
                    'description': '',
                    'confidence': 0.0
                }
            ] * len(images)
        
        results = []
        for i, image in enumerate(images):
            prompt = prompts[i] if prompts and i < len(prompts) else None
            result = self.generate_anomaly_description(image, prompt)
            results.append(result)
        
        return results

# Streamlit用のヘルパー関数
def create_streamlit_text_generation_component():
    """Streamlit用のテキスト生成コンポーネント"""
    import streamlit as st
    
    # セッション状態でテキスト生成統合を管理
    if 'text_generation' not in st.session_state:
        st.session_state.text_generation = UITextGenerationIntegration()
    
    text_gen = st.session_state.text_generation
    
    # 初期化チェック
    if not text_gen.is_initialized:
        with st.spinner("テキスト生成モデル初期化中..."):
            success = text_gen.initialize()
            if success:
                st.success("✅ テキスト生成機能が利用可能です")
            else:
                st.error("❌ テキスト生成機能の初期化に失敗しました")
                return None
    
    return text_gen

# Gradio用のヘルパー関数
def create_gradio_text_generation_interface():
    """Gradio用のテキスト生成インターフェース"""
    text_gen = UITextGenerationIntegration()
    
    # 初期化
    if not text_gen.initialize():
        logger.error("Gradio用テキスト生成機能の初期化に失敗")
        return None
    
    def generate_for_gradio(image, prompt):
        """Gradio用の生成関数"""
        if image is None:
            return "画像を選択してください"
        
        result = text_gen.generate_anomaly_description(image, prompt)
        
        if result['success']:
            return f"異常説明: {result['description']}\n確信度: {result['confidence']:.2f}"
        else:
            return f"エラー: {result['error']}"
    
    return generate_for_gradio

# テスト用関数
def test_ui_integration():
    """UI統合機能のテスト"""
    integration = UITextGenerationIntegration()
    
    # 初期化テスト
    print("=== 初期化テスト ===")
    success = integration.initialize()
    print(f"初期化結果: {success}")
    
    if success:
        # モデル状態確認
        status = integration.get_model_status()
        print(f"モデル状態: {status}")
        
        # プロンプト提案テスト
        suggestions = integration.get_generation_suggestions("設備故障")
        print(f"プロンプト提案: {suggestions[:2]}")
        
        # ダミー画像での生成テスト
        dummy_image = Image.new('RGB', (512, 512), color='blue')
        result = integration.generate_anomaly_description(dummy_image)
        print(f"生成テスト: {result}")

if __name__ == "__main__":
    test_ui_integration()
