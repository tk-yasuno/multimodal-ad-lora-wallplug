"""
FODD Pipeline: Full Online Data Description
完全オンラインデータ記述システム

新規画像に対して以下の処理を実行：
1. 画像から特徴量抽出
2. Feature Knowledge Base (FKB) での類似検索
3. LoRAテキスト生成による異常説明
4. 結果をJSON形式で保存・通知連携
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import cv2
from PIL import Image
import yaml
import sys

# プロジェクト内モジュール
# sys.path.append(str(Path(__file__).parent))

try:
    from src.knowledge_base.knowledge_manager import KnowledgeBaseManager
    from src.models.autoencoder import AnomalyDetector, ConvAutoencoder
    from src.data.preprocess import ImagePreprocessor
    from src.lora.multimodal_model import LoRAMultimodalModel
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    # 代替設定
    logger = logging.getLogger(__name__)

# ログ設定
try:
    logger = setup_logger('fodd_pipeline')
except:
    logger = logging.getLogger(__name__)


class FODDPipeline:
    """
    FODD (Full Online Data Description) パイプライン
    新規画像に対する即座の異常検知・説明生成システム
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self.load_config()
        self.initialize_components()
        
        logger.info("FODD Pipeline initialized successfully")
    
    def load_config(self):
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def initialize_components(self):
        """各コンポーネントの初期化"""
        try:
            # 画像前処理器
            self.preprocessor = ImagePreprocessor(self.config_path)
            
            # 異常検知器
            self.anomaly_detector = None
            self.load_anomaly_detector()
            
            # Knowledge Base Manager
            self.knowledge_manager = KnowledgeBaseManager(self.config)
            
            # マルチモーダル説明生成器
            self.text_generator = None
            self.load_text_generator()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_anomaly_detector(self):
        """異常検知器の読み込み"""
        try:
            model_config = self.config.get('models', {}).get('autoencoder', {})
            model_path = Path("models/autoencoder_best.pth")
            
            if model_path.exists():
                # 学習済みモデルを読み込み
                model = ConvAutoencoder(
                    input_channels=model_config.get('input_channels', 3),
                    latent_dim=model_config.get('latent_dim', 256)
                )
                
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                
                self.anomaly_detector = AnomalyDetector(
                    model=model,
                    threshold=model_config.get('anomaly_threshold', 0.1)
                )
                
                logger.info(f"Anomaly detector loaded from {model_path}")
            else:
                logger.warning("No trained anomaly detection model found")
                
        except Exception as e:
            logger.error(f"Error loading anomaly detector: {e}")
    
    def load_text_generator(self):
        """テキスト生成器の読み込み"""
        try:
            lora_config = self.config.get('lora', {})
            model_path = Path("models/lora_model")
            
            if model_path.exists():
                # LoRAモデルを使用
                text_gen_config = {
                    'text_generation': {
                        'model_name': lora_config.get('base_model', 'Salesforce/blip-image-captioning-base'),
                        'model_path': str(model_path),
                        'max_length': lora_config.get('max_length', 150),
                        'use_lora': True
                    }
                }
                self.text_generator = LoRAMultimodalModel(text_gen_config)
                logger.info("LoRA text generator loaded successfully")
            else:
                logger.warning("No LoRA model found, using base configuration")
                # フォールバック: 基本設定
                text_gen_config = {
                    'text_generation': {
                        'model_name': lora_config.get('base_model', 'Salesforce/blip-image-captioning-base'),
                        'model_path': 'models/base_model',
                        'max_length': 150,
                        'use_lora': False
                    }
                }
                self.text_generator = LoRAMultimodalModel(text_gen_config)
                
        except Exception as e:
            logger.error(f"Error loading text generator: {e}")
            self.text_generator = None
    
    def extract_image_features(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        画像から特徴量を抽出
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            特徴量ベクトルと画像メタデータ
        """
        try:
            # 画像読み込み
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # 基本メタデータ
            metadata = {
                'file_path': str(image_path),
                'file_size': Path(image_path).stat().st_size,
                'image_shape': image_array.shape,
                'processed_at': datetime.now().isoformat()
            }
            
            # 異常検知器による特徴量抽出
            if self.anomaly_detector:
                # 前処理
                processed_image = self.preprocessor.preprocess_single_image(image_path)
                processed_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
                
                # エンコーダーから特徴量抽出
                with torch.no_grad():
                    features = self.anomaly_detector.model.encoder(processed_tensor)
                    feature_vector = features.flatten().numpy()
                
                # 異常スコア計算
                anomaly_score = self.anomaly_detector.detect_anomaly(processed_tensor)
                metadata['anomaly_score'] = float(anomaly_score)
                metadata['is_anomaly'] = anomaly_score > self.anomaly_detector.threshold
                
                return feature_vector, metadata
            else:
                # フォールバック: 基本的な画像特徴量
                logger.warning("No anomaly detector available, using basic features")
                basic_features = self._extract_basic_features(image_array)
                return basic_features, metadata
                
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            raise
    
    def _extract_basic_features(self, image_array: np.ndarray) -> np.ndarray:
        """基本的な画像特徴量を抽出（フォールバック）"""
        try:
            # 色彩統計
            mean_rgb = np.mean(image_array, axis=(0, 1))
            std_rgb = np.std(image_array, axis=(0, 1))
            
            # エッジ特徴
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # テクスチャ特徴（ヒストグラム）
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_features = hist.flatten()[:50]  # 上位50bin
            
            # 特徴量結合
            features = np.concatenate([
                mean_rgb, std_rgb, [edge_density], hist_features
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            return np.zeros(256)  # デフォルト特徴量
    
    def search_similar_cases(
        self, 
        feature_vector: np.ndarray, 
        top_k: int = 5, 
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Feature Knowledge Base で類似事例を検索
        
        Args:
            feature_vector: 特徴量ベクトル
            top_k: 上位K件
            similarity_threshold: 類似度閾値
            
        Returns:
            類似事例リスト
        """
        try:
            # 特徴量ベクトルを説明テキストに変換（簡易版）
            feature_description = self._feature_vector_to_text(feature_vector)
            
            # FKBで類似検索
            similar_cases = self.knowledge_manager.search_similar_features(
                query=feature_description,
                similarity_threshold=similarity_threshold,
                max_results=top_k
            )
            
            logger.info(f"Found {len(similar_cases)} similar cases")
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error searching similar cases: {e}")
            return []
    
    def _feature_vector_to_text(self, feature_vector: np.ndarray) -> str:
        """特徴量ベクトルを検索用テキストに変換"""
        try:
            # 特徴量の統計的性質から説明を生成
            mean_val = np.mean(feature_vector)
            std_val = np.std(feature_vector)
            max_val = np.max(feature_vector)
            min_val = np.min(feature_vector)
            
            # 簡易的な特徴量説明
            descriptions = []
            
            if mean_val > 0.5:
                descriptions.append("高い活性化パターン")
            elif mean_val < -0.5:
                descriptions.append("低い活性化パターン")
            
            if std_val > 1.0:
                descriptions.append("多様な特徴分布")
            elif std_val < 0.1:
                descriptions.append("均一な特徴分布")
            
            if max_val > 2.0:
                descriptions.append("強い特徴応答")
            
            return " ".join(descriptions) if descriptions else "一般的な特徴パターン"
            
        except Exception as e:
            logger.error(f"Error converting feature vector to text: {e}")
            return "特徴量変換エラー"
    
    def generate_anomaly_description(
        self, 
        image_path: str, 
        similar_cases: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> str:
        """
        異常説明テキストを生成
        
        Args:
            image_path: 画像パス
            similar_cases: 類似事例
            metadata: 画像メタデータ
            
        Returns:
            生成された説明テキスト
        """
        try:
            if self.text_generator:
                # LoRAモデルを使用した説明生成
                image = Image.open(image_path)
                
                # 類似事例からコンテキストを構築
                context = self._build_context_from_similar_cases(similar_cases)
                
                # プロンプト構築
                prompt = self._build_explanation_prompt(metadata, context)
                
                # モデルの読み込み（必要に応じて）
                if not hasattr(self.text_generator, 'model') or self.text_generator.model is None:
                    self.text_generator.load_model()
                
                # テキスト生成
                description = self.text_generator.generate_description(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=150
                )
                
                return description
            else:
                # フォールバック: ルールベース説明
                return self._generate_rule_based_description(metadata, similar_cases)
                
        except Exception as e:
            logger.error(f"Error generating anomaly description: {e}")
            return self._generate_fallback_description(metadata)
    
    def _build_context_from_similar_cases(self, similar_cases: List[Dict[str, Any]]) -> str:
        """類似事例からコンテキストを構築"""
        if not similar_cases:
            return "類似事例なし"
        
        context_parts = []
        for case in similar_cases[:3]:  # 上位3件
            similarity = case.get('similarity', 0)
            description = case.get('description', '不明')
            context_parts.append(f"類似度{similarity:.2f}: {description}")
        
        return "; ".join(context_parts)
    
    def _build_explanation_prompt(self, metadata: Dict[str, Any], context: str) -> str:
        """説明生成用プロンプト構築"""
        anomaly_score = metadata.get('anomaly_score', 0)
        is_anomaly = metadata.get('is_anomaly', False)
        
        if is_anomaly:
            prompt = f"この画像に異常が検出されました（スコア: {anomaly_score:.3f}）。"
            if context != "類似事例なし":
                prompt += f" 類似事例: {context}。"
            prompt += " 異常の詳細を説明してください。"
        else:
            prompt = "この画像は正常と判定されました。画像の特徴を説明してください。"
        
        return prompt
    
    def _generate_rule_based_description(
        self, 
        metadata: Dict[str, Any], 
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """ルールベース説明生成（フォールバック）"""
        anomaly_score = metadata.get('anomaly_score', 0)
        is_anomaly = metadata.get('is_anomaly', False)
        
        if is_anomaly:
            severity = "高" if anomaly_score > 0.8 else "中" if anomaly_score > 0.5 else "低"
            description = f"異常検出: 異常度{severity}（スコア: {anomaly_score:.3f}）"
            
            if similar_cases:
                top_case = similar_cases[0]
                description += f"。類似事例: {top_case.get('description', '不明')}"
            
            return description
        else:
            return f"正常画像: 異常スコア{anomaly_score:.3f}（閾値以下）"
    
    def _generate_fallback_description(self, metadata: Dict[str, Any]) -> str:
        """最終フォールバック説明"""
        return f"画像処理完了（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）"
    
    def save_anomaly_report(
        self, 
        report_data: Dict[str, Any], 
        output_path: str = None
    ) -> str:
        """
        異常レポートをJSONファイルに保存
        
        Args:
            report_data: レポートデータ
            output_path: 出力パス（Noneの場合は自動生成）
            
        Returns:
            保存されたファイルパス
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"data/reports/anomaly_report_{timestamp}.json"
            
            # ディレクトリ作成
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # JSON保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Anomaly report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving anomaly report: {e}")
            raise
    
    def send_notification(self, report_data: Dict[str, Any]):
        """
        通知送信（Slack等）
        
        Args:
            report_data: レポートデータ
        """
        try:
            # 通知設定確認
            notification_config = self.config.get('notification', {})
            
            if notification_config.get('enabled', False):
                # ここで実際の通知処理を実装
                # 例: Slack Webhook, メール送信など
                logger.info("Notification sent (mock implementation)")
            else:
                logger.info("Notification disabled in configuration")
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        単一画像の完全処理パイプライン
        
        Args:
            image_path: 処理する画像のパス
            
        Returns:
            処理結果の完全レポート
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing image: {image_path}")
            
            # 1. 特徴量抽出
            feature_vector, metadata = self.extract_image_features(image_path)
            
            # 2. 類似事例検索
            similar_cases = self.search_similar_cases(feature_vector)
            
            # 3. 説明生成
            description = self.generate_anomaly_description(
                image_path, similar_cases, metadata
            )
            
            # 4. レポート構築
            report_data = {
                'timestamp': start_time.isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'image_info': {
                    'path': str(image_path),
                    'metadata': metadata
                },
                'anomaly_detection': {
                    'is_anomaly': metadata.get('is_anomaly', False),
                    'anomaly_score': metadata.get('anomaly_score', 0.0),
                    'threshold': getattr(self.anomaly_detector, 'threshold', 0.1) if self.anomaly_detector else 0.1
                },
                'similar_cases': similar_cases,
                'generated_description': description,
                'feature_vector_shape': feature_vector.shape,
                'system_info': {
                    'fodd_version': '1.0.0',
                    'config_path': self.config_path
                }
            }
            
            # 5. レポート保存
            report_path = self.save_anomaly_report(report_data)
            report_data['report_path'] = report_path
            
            # 6. 通知送信
            if metadata.get('is_anomaly', False):
                self.send_notification(report_data)
            
            logger.info(f"Image processing completed: {image_path}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            
            # エラーレポート
            error_report = {
                'timestamp': start_time.isoformat(),
                'error': str(e),
                'image_path': str(image_path),
                'status': 'failed'
            }
            
            return error_report
    
    def process_batch_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        複数画像のバッチ処理
        
        Args:
            image_paths: 処理する画像パスのリスト
            
        Returns:
            各画像の処理結果リスト
        """
        results = []
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_single_image(image_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def close(self):
        """リソースのクリーンアップ"""
        try:
            if hasattr(self, 'knowledge_manager'):
                self.knowledge_manager.close()
            
            logger.info("FODD Pipeline closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing FODD Pipeline: {e}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FODD Pipeline - Full Online Data Description")
    parser.add_argument("--image", type=str, help="Single image path to process")
    parser.add_argument("--batch", type=str, help="Directory containing images for batch processing")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--output", type=str, help="Output report path")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with sample data")
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_logger("fodd_pipeline")
    
    try:
        # FODD Pipeline初期化
        pipeline = FODDPipeline(args.config)
        
        if args.demo:
            # デモモード: サンプル画像でテスト
            logger.info("Running in demo mode")
            sample_image = "data/images/normal/sample_001.jpg"
            
            if Path(sample_image).exists():
                result = pipeline.process_single_image(sample_image)
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                logger.error(f"Sample image not found: {sample_image}")
                
        elif args.image:
            # 単一画像処理
            result = pipeline.process_single_image(args.image)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Result saved to: {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                
        elif args.batch:
            # バッチ処理
            image_dir = Path(args.batch)
            if not image_dir.exists():
                logger.error(f"Batch directory not found: {image_dir}")
                return
            
            # 画像ファイル検索
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(image_dir.glob(f"**/*{ext}"))
                image_paths.extend(image_dir.glob(f"**/*{ext.upper()}"))
            
            if not image_paths:
                logger.error(f"No images found in: {image_dir}")
                return
            
            # バッチ処理実行
            results = pipeline.process_batch_images([str(p) for p in image_paths])
            
            # 結果保存
            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"data/reports/batch_anomaly_report_{timestamp}.json"
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Batch results saved to: {output_path}")
            
        else:
            logger.error("Please specify --image, --batch, or --demo")
            parser.print_help()
            
    except Exception as e:
        logger.error(f"FODD Pipeline execution failed: {e}")
        raise
    
    finally:
        # リソースクリーンアップ
        if 'pipeline' in locals():
            pipeline.close()


if __name__ == "__main__":
    main()
