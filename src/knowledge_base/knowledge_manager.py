"""
Knowledge Base Manager - ナレッジベースの統合管理
異常検知システムとの連携、特徴量抽出、推論支援
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from datetime import datetime
import json

from .vector_database import FeatureKnowledgeBase
from ..ui.feedback_manager import FeedbackDataManager

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    ナレッジベース統合管理クラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb_config = config.get('knowledge_base', {})
        
        # Knowledge Base初期化
        kb_path = self.kb_config.get('db_path', 'data/knowledge_base')
        self.knowledge_base = FeatureKnowledgeBase(kb_path)
        
        # フィードバックマネージャー初期化
        feedback_db_path = config.get('feedback', {}).get('db_path', 'data/feedback/feedback.db')
        self.feedback_manager = FeedbackDataManager(feedback_db_path)
        
        logger.info("Knowledge Base Manager 初期化完了")
    
    def extract_features_from_feedback(self, min_confidence: float = 3.0) -> int:
        """
        フィードバックデータから特徴量を抽出してナレッジベースに追加
        
        Args:
            min_confidence: 最小信頼度スコア
            
        Returns:
            追加されたレコード数
        """
        try:
            # フィードバックデータを取得
            feedback_data = self.feedback_manager.get_feedback_data()
            
            added_count = 0
            for data in feedback_data:
                # 信頼度フィルタ
                if data.get('confidence_level', 0) < min_confidence:
                    continue
                
                # 異常データのみ処理
                if not data.get('is_anomaly', False):
                    continue
                
                # 説明文が存在するもののみ
                description = data.get('anomaly_description', '').strip()
                if not description:
                    continue
                
                # ナレッジベースに追加
                record_id = self.knowledge_base.add_feature(
                    image_path=data.get('image_path', ''),
                    anomaly_type=data.get('anomaly_type', '不明'),
                    anomaly_description=description,
                    confidence_score=data.get('confidence_level', 0),
                    metadata={
                        'source': 'feedback',
                        'feedback_id': data.get('id'),
                        'extracted_at': datetime.now().isoformat()
                    }
                )
                
                if record_id:
                    added_count += 1
                    logger.debug(f"特徴量追加: {record_id}")
            
            logger.info(f"フィードバックデータから{added_count}件の特徴量を抽出")
            return added_count
            
        except Exception as e:
            logger.error(f"特徴量抽出エラー: {e}")
            return 0
    
    def find_similar_cases(
        self,
        anomaly_description: str,
        anomaly_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        類似事例検索
        
        Args:
            anomaly_description: 異常の説明
            anomaly_type: 異常タイプでのフィルタ（オプション）
            top_k: 取得する上位件数
            
        Returns:
            類似事例のリスト
        """
        try:
            results = self.knowledge_base.search_similar_features(
                query_text=anomaly_description,
                k=top_k,
                anomaly_type_filter=anomaly_type
            )
            
            logger.info(f"類似事例検索完了: {len(results)}件")
            return results
            
        except Exception as e:
            logger.error(f"類似事例検索エラー: {e}")
            return []
    
    def suggest_anomaly_description(
        self,
        image_path: str,
        detected_anomaly_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        異常説明の提案
        
        Args:
            image_path: 画像パス
            detected_anomaly_type: 検出された異常タイプ
            context: 追加コンテキスト
            
        Returns:
            提案された説明と関連情報
        """
        try:
            # 同じ異常タイプの類似事例を検索
            query_text = f"{detected_anomaly_type} 異常"
            similar_cases = self.knowledge_base.search_similar_features(
                query_text=query_text,
                k=3,
                anomaly_type_filter=detected_anomaly_type
            )
            
            suggestions = []
            confidence_scores = []
            
            for case in similar_cases:
                suggestions.append(case['anomaly_description'])
                confidence_scores.append(case['confidence_score'])
            
            # 最も一般的な表現パターンを抽出
            if suggestions:
                # 単純化: 最も高い信頼度の説明を基本テンプレートとする
                best_idx = np.argmax(confidence_scores)
                base_description = suggestions[best_idx]
                
                # パターン分析（簡易版）
                common_phrases = self._extract_common_phrases(suggestions)
                
                result = {
                    'suggested_description': base_description,
                    'alternative_descriptions': suggestions[:3],
                    'common_phrases': common_phrases,
                    'confidence': np.mean(confidence_scores),
                    'similar_cases_count': len(similar_cases),
                    'anomaly_type': detected_anomaly_type
                }
            else:
                # 類似事例がない場合のデフォルト提案
                result = {
                    'suggested_description': f"{detected_anomaly_type}が検出されました。詳細な確認が必要です。",
                    'alternative_descriptions': [],
                    'common_phrases': [],
                    'confidence': 0.0,
                    'similar_cases_count': 0,
                    'anomaly_type': detected_anomaly_type
                }
            
            logger.info(f"異常説明提案完了: {detected_anomaly_type}")
            return result
            
        except Exception as e:
            logger.error(f"異常説明提案エラー: {e}")
            return {}
    
    def _extract_common_phrases(self, descriptions: List[str]) -> List[str]:
        """共通フレーズ抽出（簡易版）"""
        try:
            # 単語レベルでの共通性分析
            word_counts = {}
            for desc in descriptions:
                words = desc.split()
                for word in words:
                    if len(word) > 2:  # 2文字以上の単語のみ
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # 出現頻度の高い単語を抽出
            common_words = [
                word for word, count in word_counts.items() 
                if count >= len(descriptions) * 0.3  # 30%以上の説明に出現
            ]
            
            return common_words[:10]  # 上位10個
            
        except Exception as e:
            logger.error(f"共通フレーズ抽出エラー: {e}")
            return []
    
    def get_anomaly_patterns(self, anomaly_type: Optional[str] = None) -> Dict[str, Any]:
        """
        異常パターン分析
        
        Args:
            anomaly_type: 特定の異常タイプでフィルタ
            
        Returns:
            異常パターンの分析結果
        """
        try:
            stats = self.knowledge_base.get_statistics()
            
            # 全体統計
            result = {
                'total_records': stats.get('total_records', 0),
                'anomaly_types': stats.get('anomaly_types', {}),
                'latest_update': stats.get('latest_timestamp'),
            }
            
            # 特定タイプの詳細分析
            if anomaly_type and anomaly_type in stats.get('anomaly_types', {}):
                similar_cases = self.knowledge_base.search_similar_features(
                    query_text=anomaly_type,
                    k=50,
                    anomaly_type_filter=anomaly_type
                )
                
                # 信頼度分布
                confidence_scores = [case['confidence_score'] for case in similar_cases]
                if confidence_scores:
                    result['type_analysis'] = {
                        'anomaly_type': anomaly_type,
                        'count': len(similar_cases),
                        'avg_confidence': np.mean(confidence_scores),
                        'min_confidence': np.min(confidence_scores),
                        'max_confidence': np.max(confidence_scores),
                        'common_descriptions': [
                            case['anomaly_description'] for case in similar_cases[:5]
                        ]
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"異常パターン分析エラー: {e}")
            return {}
    
    def update_knowledge_base(self, force_update: bool = False) -> Dict[str, int]:
        """
        ナレッジベースの更新
        
        Args:
            force_update: 強制更新フラグ
            
        Returns:
            更新結果の統計
        """
        try:
            logger.info("ナレッジベース更新開始")
            
            # フィードバックデータから特徴量抽出
            extracted_count = self.extract_features_from_feedback()
            
            # データベース保存
            self.knowledge_base.save()
            
            # 統計情報更新
            stats = self.knowledge_base.get_statistics()
            
            result = {
                'extracted_features': extracted_count,
                'total_records': stats.get('total_records', 0),
                'vector_db_size': stats.get('vector_db_size', 0)
            }
            
            logger.info(f"ナレッジベース更新完了: {result}")
            return result
            
        except Exception as e:
            logger.error(f"ナレッジベース更新エラー: {e}")
            return {}
    
    def export_knowledge_summary(self, output_path: str) -> bool:
        """
        ナレッジベースサマリーのエクスポート
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            エクスポート成功フラグ
        """
        try:
            stats = self.knowledge_base.get_statistics()
            
            summary = {
                'export_date': datetime.now().isoformat(),
                'statistics': stats,
                'anomaly_patterns': {}
            }
            
            # 各異常タイプのパターン分析
            for anomaly_type in stats.get('anomaly_types', {}):
                patterns = self.get_anomaly_patterns(anomaly_type)
                summary['anomaly_patterns'][anomaly_type] = patterns.get('type_analysis', {})
            
            # JSONファイルに出力
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ナレッジベースサマリー出力完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"サマリー出力エラー: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """ナレッジベース統計情報取得"""
        try:
            stats = {}
            
            # 基本統計
            total_count = self.knowledge_base.get_total_count()
            stats['total_features'] = total_count
            stats['vector_dimension'] = self.knowledge_base.vector_db.dimension
            
            # カテゴリ別統計
            category_stats = self.knowledge_base.get_category_stats()
            stats['category_distribution'] = category_stats
            stats['normal_count'] = category_stats.get('normal', 0)
            stats['anomaly_count'] = category_stats.get('anomaly', 0)
            
            # データベース情報
            stats['database_info'] = {
                'db_path': str(self.knowledge_base.db_path),
                'vector_index_type': self.knowledge_base.vector_db.index_type,
                'embedding_model': self.knowledge_base.text_embedder.model_name
            }
            
            # 時系列分布（過去30日）
            temporal_stats = self.knowledge_base.get_temporal_distribution(days=30)
            stats['temporal_distribution'] = temporal_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {}
    
    def search_similar_features(
        self,
        query: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """類似特徴量検索"""
        try:
            results = self.knowledge_base.search_similar(
                query_text=query,
                top_k=max_results,
                threshold=similarity_threshold
            )
            
            return [
                {
                    'id': result.id,
                    'description': result.description,
                    'category': result.category,
                    'confidence': result.confidence,
                    'source': result.source,
                    'metadata': result.metadata,
                    'similarity': result.similarity if hasattr(result, 'similarity') else 0.0,
                    'created_at': result.created_at.isoformat() if result.created_at else None
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"類似特徴量検索エラー: {e}")
            return []
    
    def add_feature(
        self,
        description: str,
        category: str = "unknown",
        confidence: float = 0.8,
        source: str = "manual",
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """特徴量手動追加"""
        try:
            from src.knowledge_base.vector_database import FeatureRecord
            
            record = FeatureRecord(
                description=description,
                category=category,
                confidence=confidence,
                source=source,
                metadata=metadata or {}
            )
            
            feature_id = self.knowledge_base.add_feature(record)
            logger.info(f"特徴量手動追加: {feature_id}")
            return feature_id
            
        except Exception as e:
            logger.error(f"特徴量追加エラー: {e}")
            return None
    
    def analyze_anomaly_patterns(self, threshold: float = 0.8) -> List[str]:
        """異常パターン分析"""
        try:
            patterns = []
            
            # 高信頼度の異常データを取得
            anomaly_features = self.knowledge_base.get_features_by_category(
                category="anomaly",
                min_confidence=threshold
            )
            
            # パターン分析
            from collections import Counter
            
            # 頻出キーワード分析
            all_descriptions = [f.description for f in anomaly_features]
            keywords = []
            for desc in all_descriptions:
                words = desc.lower().split()
                keywords.extend([w for w in words if len(w) > 2])
            
            common_keywords = Counter(keywords).most_common(5)
            
            for keyword, count in common_keywords:
                if count > 1:
                    patterns.append(f"キーワード '{keyword}' が {count} 回出現")
            
            # メタデータパターン分析
            metadata_patterns = {}
            for feature in anomaly_features:
                if feature.metadata:
                    for key, value in feature.metadata.items():
                        if key not in metadata_patterns:
                            metadata_patterns[key] = Counter()
                        metadata_patterns[key][str(value)] += 1
            
            for key, value_counts in metadata_patterns.items():
                for value, count in value_counts.most_common(3):
                    if count > 1:
                        patterns.append(f"{key}: {value} が {count} 回出現")
            
            return patterns
            
        except Exception as e:
            logger.error(f"異常パターン分析エラー: {e}")
            return []
    
    def analyze_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度分析"""
        try:
            importance_scores = {}
            
            # カテゴリ別の出現頻度から重要度を計算
            all_features = self.knowledge_base.get_all_features()
            
            # 単語レベルの重要度
            word_counts = {}
            total_words = 0
            
            for feature in all_features:
                words = feature.description.lower().split()
                for word in words:
                    if len(word) > 2:  # 短い単語は除外
                        word_counts[word] = word_counts.get(word, 0) + 1
                        total_words += 1
            
            # TF-IDF風の重要度計算
            for word, count in word_counts.items():
                tf = count / total_words
                importance_scores[word] = tf * feature.confidence if hasattr(feature, 'confidence') else tf
            
            # 上位10位を返す
            sorted_scores = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10])
            return sorted_scores
            
        except Exception as e:
            logger.error(f"重要度分析エラー: {e}")
            return {}
    
    def export_knowledge_base(
        self,
        format: str = "json",
        category_filter: Optional[List[str]] = None,
        confidence_min: float = 0.0,
        include_vectors: bool = False
    ) -> Optional[str]:
        """ナレッジベースエクスポート"""
        try:
            # フィルタリングして特徴量取得
            all_features = self.knowledge_base.get_all_features()
            
            filtered_features = []
            for feature in all_features:
                # カテゴリフィルタ
                if category_filter and feature.category not in category_filter:
                    continue
                
                # 信頼度フィルタ
                if feature.confidence < confidence_min:
                    continue
                
                filtered_features.append(feature)
            
            # エクスポートデータ構築
            export_data = []
            for feature in filtered_features:
                data = {
                    'id': feature.id,
                    'description': feature.description,
                    'category': feature.category,
                    'confidence': feature.confidence,
                    'source': feature.source,
                    'metadata': feature.metadata,
                    'created_at': feature.created_at.isoformat() if feature.created_at else None
                }
                
                if include_vectors and hasattr(feature, 'vector'):
                    data['vector'] = feature.vector.tolist() if feature.vector is not None else None
                
                export_data.append(data)
            
            # フォーマット別出力
            if format == "json":
                import json
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            
            elif format == "csv":
                import csv
                import io
                
                output = io.StringIO()
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
                
                return output.getvalue()
            
            else:
                # その他のフォーマットは未実装
                logger.warning(f"未対応のエクスポート形式: {format}")
                return None
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {e}")
            return None
    
    def close(self):
        """リソースのクリーンアップ"""
        try:
            self.knowledge_base.close()
            logger.info("Knowledge Base Manager クローズ完了")
        except Exception as e:
            logger.error(f"クローズエラー: {e}")
