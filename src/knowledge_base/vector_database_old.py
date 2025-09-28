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
    """
    FAISSベースのベクトルデータベース
    """
    
    def __init__(self, dimension: int = 512, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.records: List[FeatureRecord] = []
        self.id_to_idx: Dict[str, int] = {}
        
        self._initialize_index()
        
    def _initialize_index(self):
        """FAISSインデックスの初期化"""
        if self.index_type == "IVF":
            # IVF (Inverted File) インデックス
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) インデックス
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # フラットインデックス（デフォルト）
            self.index = faiss.IndexFlatL2(self.dimension)
            
        logger.info(f"FAISSインデックス初期化完了: {self.index_type}, 次元数: {self.dimension}")
    
    def add_record(self, record: FeatureRecord) -> bool:
        """レコードを追加"""
        try:
            # ベクトルの次元チェック
            if record.feature_vector.shape[0] != self.dimension:
                logger.error(f"ベクトルの次元が不正: {record.feature_vector.shape[0]} != {self.dimension}")
                return False
            
            # IVFインデックスの場合は学習が必要
            if self.index_type == "IVF" and not self.index.is_trained:
                if len(self.records) >= 100:  # 最低100個のベクトルで学習
                    self._train_index()
            
            # インデックスに追加
            vector = record.feature_vector.reshape(1, -1).astype(np.float32)
            self.index.add(vector)
            
            # メタデータを保存
            idx = len(self.records)
            self.records.append(record)
            self.id_to_idx[record.id] = idx
            
            logger.debug(f"レコード追加完了: {record.id}")
            return True
            
        except Exception as e:
            logger.error(f"レコード追加エラー: {e}")
            return False
    
    def _train_index(self):
        """IVFインデックスの学習"""
        if len(self.records) < 100:
            logger.warning("学習用データが不足しています")
            return
            
        vectors = np.array([record.feature_vector for record in self.records]).astype(np.float32)
        self.index.train(vectors)
        logger.info("IVFインデックスの学習完了")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[FeatureRecord, float]]:
        """類似ベクトル検索"""
        try:
            if len(self.records) == 0:
                return []
            
            # ベクトルの次元チェック
            if query_vector.shape[0] != self.dimension:
                logger.error(f"クエリベクトルの次元が不正: {query_vector.shape[0]} != {self.dimension}")
                return []
            
            # 検索実行
            query = query_vector.reshape(1, -1).astype(np.float32)
            distances, indices = self.index.search(query, min(k, len(self.records)))
            
            # 結果を整形
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.records):
                    results.append((self.records[idx], float(dist)))
            
            return results
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []
    
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
            
            with open(filepath / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # ベクトルデータを保存
            vectors = np.array([record.feature_vector for record in self.records])
            np.save(filepath / "vectors.npy", vectors)
            
            logger.info(f"ベクトルデータベース保存完了: {filepath}")
            
        except Exception as e:
            logger.error(f"保存エラー: {e}")
    
    def load(self, filepath: str) -> bool:
        """インデックスとメタデータを読み込み"""
        try:
            filepath = Path(filepath)
            
            # FAISSインデックスを読み込み
            index_path = filepath / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # メタデータを読み込み
            metadata_path = filepath / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.dimension = metadata['dimension']
                self.index_type = metadata['index_type']
                self.id_to_idx = metadata['id_to_idx']
            
            # ベクトルデータを読み込み
            vectors_path = filepath / "vectors.npy"
            if vectors_path.exists():
                vectors = np.load(vectors_path)
                
                # レコードを再構築
                for i, (record_data, vector) in enumerate(zip(metadata['records'], vectors)):
                    record = FeatureRecord(
                        id=record_data['id'],
                        image_path=record_data['image_path'],
                        feature_vector=vector,
                        anomaly_type=record_data['anomaly_type'],
                        anomaly_description=record_data['anomaly_description'],
                        confidence_score=record_data['confidence_score'],
                        timestamp=record_data['timestamp'],
                        metadata=record_data['metadata']
                    )
                    self.records.append(record)
            
            logger.info(f"ベクトルデータベース読み込み完了: {len(self.records)}件")
            return True
            
        except Exception as e:
            logger.error(f"読み込みエラー: {e}")
            return False


class TextEmbedder:
    """
    テキスト埋め込みベクトル生成クラス
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """モデルの読み込み"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"テキスト埋め込みモデル読み込み完了: {self.model_name}")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """テキストをベクトル化"""
        try:
            if self.model is None:
                logger.error("モデルが読み込まれていません")
                return np.array([])
            
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"ベクトル化エラー: {e}")
            return np.array([])
    
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
    
    def _init_sql_db(self):
        """SQLiteデータベースの初期化"""
        try:
            with sqlite3.connect(self.sql_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS features (
                        id TEXT PRIMARY KEY,
                        image_path TEXT,
                        anomaly_type TEXT,
                        anomaly_description TEXT,
                        confidence_score REAL,
                        timestamp TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anomaly_type ON features(anomaly_type)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp)
                """)
                
        except Exception as e:
            logger.error(f"SQLite初期化エラー: {e}")
    
    def _load_existing_data(self):
        """既存データの読み込み"""
        vector_db_path = self.db_path / "vectors"
        if vector_db_path.exists():
            except Exception as e:
                logger.warning(f"ベクトルデータベース読み込み警告: {e}")
        except Exception as e:
            logger.warning(f"既存データ読み込み警告: {e}")
    
    def add_feature(
        self,
        image_path: str,
        anomaly_type: str,
        anomaly_description: str,
        confidence_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """新しい特徴量レコードを追加"""
        try:
            # レコードID生成
            record_id = f"feat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # テキスト埋め込みベクトル生成
            combined_text = f"{anomaly_type}: {anomaly_description}"
            feature_vector = self.text_embedder.encode_single(combined_text)
            
            if len(feature_vector) == 0:
                logger.error("特徴量ベクトルの生成に失敗")
                return ""
            
            # レコード作成
            record = FeatureRecord(
                id=record_id,
                image_path=image_path,
                feature_vector=feature_vector,
                anomaly_type=anomaly_type,
                anomaly_description=anomaly_description,
                confidence_score=confidence_score,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # ベクトルデータベースに追加
            if not self.vector_db.add_record(record):
                logger.error("ベクトルデータベースへの追加に失敗")
                return ""
            
            # SQLiteデータベースに追加
            with sqlite3.connect(self.sql_db_path) as conn:
                conn.execute("""
                    INSERT INTO features 
                    (id, image_path, anomaly_type, anomaly_description, confidence_score, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.image_path,
                    record.anomaly_type,
                    record.anomaly_description,
                    record.confidence_score,
                    record.timestamp,
                    json.dumps(record.metadata)
                ))
            
            logger.info(f"特徴量レコード追加完了: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"特徴量追加エラー: {e}")
            return ""
    
    def search_similar_features(
        self,
        query_text: str,
        k: int = 5,
        anomaly_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """類似特徴量検索"""
        try:
            # クエリテキストをベクトル化
            query_vector = self.text_embedder.encode_single(query_text)
            if len(query_vector) == 0:
                return []
            
            # ベクトル検索
            search_results = self.vector_db.search(query_vector, k * 2)  # 多めに取得してフィルタ
            
            # 結果整形
            results = []
            for record, distance in search_results:
                if anomaly_type_filter and record.anomaly_type != anomaly_type_filter:
                    continue
                
                result = {
                    'id': record.id,
                    'image_path': record.image_path,
                    'anomaly_type': record.anomaly_type,
                    'anomaly_description': record.anomaly_description,
                    'confidence_score': record.confidence_score,
                    'similarity_score': 1.0 / (1.0 + distance),  # 距離を類似度に変換
                    'timestamp': record.timestamp,
                    'metadata': record.metadata
                }
                results.append(result)
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            with sqlite3.connect(self.sql_db_path) as conn:
                cursor = conn.cursor()
                
                # 総レコード数
                cursor.execute("SELECT COUNT(*) FROM features")
                total_records = cursor.fetchone()[0]
                
                # 異常タイプ別統計
                cursor.execute("""
                    SELECT anomaly_type, COUNT(*) 
                    FROM features 
                    GROUP BY anomaly_type
                """)
                anomaly_types = dict(cursor.fetchall())
                
                # 最新レコード
                cursor.execute("""
                    SELECT timestamp 
                    FROM features 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                latest_record = cursor.fetchone()
                latest_timestamp = latest_record[0] if latest_record else None
                
                return {
                    'total_records': total_records,
                    'anomaly_types': anomaly_types,
                    'latest_timestamp': latest_timestamp,
                    'vector_db_size': len(self.vector_db.records)
                }
                
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {}
    
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
        logger.info("Feature Knowledge Base クローズ完了")
