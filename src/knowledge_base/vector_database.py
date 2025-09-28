"""
Feature Knowledge Base (FKB) - ベクトルデータベース管理
異常検知で抽出された特徴量をベクトル化して検索可能な形で保存・検索
"""

import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import sqlite3
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class FeatureRecord:
    """特徴量レコードのデータクラス"""
    description: str
    category: str = "unknown"
    confidence: float = 0.8
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    vector: Optional[np.ndarray] = None
    similarity: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """辞書形式に変換（ベクトルは除く）"""
        result = asdict(self)
        if 'vector' in result:
            result.pop('vector')  # ベクトルは別途保存
        return result


class VectorDatabase:
    """FAISSベースのベクトルデータベース"""
    
    def __init__(self, dimension: int = 384, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.records: List[FeatureRecord] = []
        self.id_to_idx: Dict[str, int] = {}
        
        # FAISSインデックス初期化
        self._init_index()
        logger.info(f"FAISSインデックス初期化完了: {index_type}, 次元数: {dimension}")
    
    def _init_index(self):
        """FAISSインデックスの初期化"""
        if self.index_type == "IVF":
            # IVF (Inverted File) インデックス
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) インデックス
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # シンプルなFlatインデックス
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_record(self, record: FeatureRecord) -> bool:
        """レコードを追加"""
        try:
            if record.vector is None or len(record.vector) != self.dimension:
                logger.error(f"無効なベクトル次元: {len(record.vector) if record.vector is not None else 'None'}")
                return False
            
            # レコードリストに追加
            idx = len(self.records)
            self.records.append(record)
            self.id_to_idx[record.id] = idx
            
            # FAISSインデックスに追加
            vector_array = record.vector.reshape(1, -1).astype('float32')
            
            if self.index_type == "IVF" and not self.index.is_trained:
                # IVFインデックスの場合、最初のデータでトレーニング
                if len(self.records) >= 100:  # 十分なデータが揃ったらトレーニング
                    training_vectors = np.array([r.vector for r in self.records[-100:]]).astype('float32')
                    self.index.train(training_vectors)
            
            if self.index.is_trained or self.index_type != "IVF":
                self.index.add(vector_array)
            
            return True
            
        except Exception as e:
            logger.error(f"レコード追加エラー: {e}")
            return False
    
    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[FeatureRecord]:
        """類似ベクトル検索"""
        try:
            if len(self.records) == 0:
                return []
            
            query_array = query_vector.reshape(1, -1).astype('float32')
            
            # 検索実行
            if self.index.is_trained or self.index_type != "IVF":
                distances, indices = self.index.search(query_array, min(k, len(self.records)))
            else:
                # インデックスがトレーニングされていない場合は線形検索
                distances, indices = self._linear_search(query_vector, k)
            
            # 結果を構築
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.records):
                    record = self.records[idx]
                    # 類似度を距離から計算（0-1の範囲）
                    similarity = 1.0 / (1.0 + distance)
                    record.similarity = similarity
                    results.append(record)
            
            return results
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []
    
    def _linear_search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """線形検索（フォールバック）"""
        distances = []
        for record in self.records:
            if record.vector is not None:
                # L2距離計算
                distance = np.linalg.norm(query_vector - record.vector)
                distances.append(distance)
            else:
                distances.append(float('inf'))
        
        # トップK要素を取得
        sorted_indices = np.argsort(distances)[:k]
        sorted_distances = [distances[i] for i in sorted_indices]
        
        return np.array([sorted_distances]), np.array([sorted_indices])
    
    def save(self, filepath: str):
        """インデックスとメタデータを保存"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # FAISSインデックスを保存
            faiss.write_index(self.index, str(filepath / "index.faiss"))
            
            # メタデータを保存
            metadata = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'records': [record.to_dict() for record in self.records],
                'id_to_idx': self.id_to_idx
            }
            
            with open(filepath / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"ベクトルデータベース保存完了: {filepath}")
            
        except Exception as e:
            logger.error(f"保存エラー: {e}")
    
    def load(self, filepath: str):
        """インデックスとメタデータを読み込み"""
        try:
            filepath = Path(filepath)
            
            # FAISSインデックスを読み込み
            index_path = filepath / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # メタデータを読み込み
            metadata_path = filepath / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.dimension = metadata.get('dimension', self.dimension)
                self.index_type = metadata.get('index_type', self.index_type)
                self.id_to_idx = metadata.get('id_to_idx', {})
                
                # レコードを復元
                for record_dict in metadata.get('records', []):
                    record = FeatureRecord(**record_dict)
                    self.records.append(record)
            
            logger.info(f"ベクトルデータベース読み込み完了: {filepath}")
            
        except Exception as e:
            logger.error(f"読み込みエラー: {e}")


class TextEmbedder:
    """テキスト埋め込みクラス"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"テキスト埋め込みモデル読み込み完了: {model_name}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """テキストリストをベクトル化"""
        return self.model.encode(texts)
    
    def encode_single(self, text: str) -> np.ndarray:
        """単一テキストをベクトル化"""
        embeddings = self.encode([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])


class FeatureKnowledgeBase:
    """
    Feature Knowledge Base メインクラス
    異常検知の特徴量とテキスト説明を統合管理
    """
    
    def __init__(self, db_path: str = "data/knowledge_base"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # テキスト埋め込みモデル
        self.text_embedder = TextEmbedder()
        embedding_dim = 384  # all-MiniLM-L6-v2の出力次元
        
        # ベクトルデータベース
        self.vector_db = VectorDatabase(dimension=embedding_dim)
        
        # SQLiteデータベース（構造化データ用）
        self.sql_db_path = self.db_path / "metadata.db"
        self._init_sql_db()
        
        # 既存データの読み込み
        self._load_existing_data()
        
        logger.info("Feature Knowledge Base 初期化完了")
    
    def _init_sql_db(self):
        """SQLiteデータベースの初期化"""
        self.conn = sqlite3.connect(str(self.sql_db_path))
        cursor = self.conn.cursor()
        
        # テーブル作成
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT
            )
        """)
        
        self.conn.commit()
        logger.info(f"SQLiteデータベース初期化完了: {self.sql_db_path}")
    
    def _load_existing_data(self):
        """既存データの読み込み"""
        try:
            vector_db_path = self.db_path / "vectors"
            if vector_db_path.exists():
                self.vector_db.load(str(vector_db_path))
                logger.info("既存ベクトルデータベース読み込み完了")
        except Exception as e:
            logger.warning(f"既存データ読み込み警告: {e}")
    
    def add_feature(self, record: FeatureRecord) -> Optional[str]:
        """新しい特徴量レコードを追加"""
        try:
            # レコードID生成
            if not record.id:
                record.id = f"feat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # テキスト埋め込みベクトル生成
            feature_vector = self.text_embedder.encode_single(record.description)
            
            if len(feature_vector) == 0:
                logger.error("特徴量ベクトルの生成に失敗")
                return None
            
            record.vector = feature_vector
            
            # ベクトルデータベースに追加
            if not self.vector_db.add_record(record):
                logger.error("ベクトルデータベースへの追加に失敗")
                return None
            
            # SQLiteに構造化データを保存
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO features (id, description, category, confidence, source, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.description,
                record.category,
                record.confidence,
                record.source,
                json.dumps(record.metadata),
                record.created_at.isoformat() if record.created_at else datetime.now().isoformat()
            ))
            
            self.conn.commit()
            
            logger.info(f"特徴量レコード追加完了: {record.id}")
            return record.id
            
        except Exception as e:
            logger.error(f"特徴量追加エラー: {e}")
            return None
    
    def search_similar(
        self, 
        query_text: str, 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[FeatureRecord]:
        """類似特徴量検索"""
        try:
            # クエリテキストをベクトル化
            query_vector = self.text_embedder.encode_single(query_text)
            
            if len(query_vector) == 0:
                logger.error("クエリベクトルの生成に失敗")
                return []
            
            # ベクトル検索実行
            similar_records = self.vector_db.search_similar(query_vector, top_k)
            
            # 閾値フィルタリング
            filtered_records = [
                record for record in similar_records 
                if getattr(record, 'similarity', 0) >= threshold
            ]
            
            return filtered_records
            
        except Exception as e:
            logger.error(f"類似検索エラー: {e}")
            return []
    
    def get_total_count(self) -> int:
        """総レコード数を取得"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM features")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"総数取得エラー: {e}")
            return 0
    
    def get_category_stats(self) -> Dict[str, int]:
        """カテゴリ別統計を取得"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM features 
                GROUP BY category
            """)
            return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"カテゴリ統計取得エラー: {e}")
            return {}
    
    def get_temporal_distribution(self, days: int = 30) -> List[Dict]:
        """時系列分布を取得"""
        try:
            from datetime import datetime, timedelta
            
            cursor = self.conn.cursor()
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM features 
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (start_date.isoformat(),))
            
            results = cursor.fetchall()
            return [{'date': row[0], 'count': row[1]} for row in results]
            
        except Exception as e:
            logger.error(f"時系列分布取得エラー: {e}")
            return []
    
    def get_features_by_category(
        self, 
        category: str, 
        min_confidence: float = 0.0
    ) -> List[FeatureRecord]:
        """カテゴリ別特徴量取得"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM features 
                WHERE category = ? AND confidence >= ?
                ORDER BY confidence DESC
            """, (category, min_confidence))
            
            features = []
            for row in cursor.fetchall():
                record = FeatureRecord(
                    id=row[0],
                    description=row[1],
                    category=row[2],
                    confidence=row[3],
                    source=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                    created_at=datetime.fromisoformat(row[6]) if row[6] else None
                )
                features.append(record)
            
            return features
            
        except Exception as e:
            logger.error(f"カテゴリ別特徴量取得エラー: {e}")
            return []
    
    def get_all_features(self) -> List[FeatureRecord]:
        """全特徴量取得"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM features ORDER BY created_at DESC")
            
            features = []
            for row in cursor.fetchall():
                record = FeatureRecord(
                    id=row[0],
                    description=row[1],
                    category=row[2],
                    confidence=row[3],
                    source=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                    created_at=datetime.fromisoformat(row[6]) if row[6] else None
                )
                features.append(record)
            
            return features
            
        except Exception as e:
            logger.error(f"全特徴量取得エラー: {e}")
            return []
    
    def save(self):
        """データベースを保存"""
        try:
            vector_db_path = self.db_path / "vectors"
            self.vector_db.save(vector_db_path)
            logger.info("Feature Knowledge Base 保存完了")
        except Exception as e:
            logger.error(f"保存エラー: {e}")
    
    def close(self):
        """リソースのクリーンアップ"""
        self.save()
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("Feature Knowledge Base クローズ完了")
