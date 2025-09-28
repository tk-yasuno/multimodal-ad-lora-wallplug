"""
MAD-FH: Human Feedback Data Manager
人間のフィードバックデータの管理・保存機能
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid


class FeedbackDataManager:
    """人間フィードバックデータの管理クラス"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLiteデータベースファイルのパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """データベースの初期化とテーブル作成"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # フィードバックテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    image_id INTEGER,
                    image_path TEXT NOT NULL,
                    is_anomaly BOOLEAN,
                    anomaly_type TEXT,
                    anomaly_description TEXT,
                    confidence_level INTEGER,
                    user_id TEXT,
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT,
                    FOREIGN KEY (image_id) REFERENCES images (id)
                )
            """)
            
            # セッションテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_feedbacks INTEGER DEFAULT 0,
                    session_metadata TEXT
                )
            """)
            
            # 異常タイプマスターテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    category TEXT,
                    severity_level INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # インデックス作成
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_image_path ON feedback(image_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_anomaly_type ON feedback(anomaly_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)")
            
            conn.commit()
            
            # デフォルトの異常タイプを追加
            self._init_default_anomaly_types()
    
    def _init_default_anomaly_types(self):
        """デフォルトの異常タイプを初期化"""
        default_types = [
            ("設備故障", "機械や設備の故障・異常", "Equipment", 5),
            ("製品欠陥", "製品の品質異常", "Product", 4),
            ("安全問題", "安全に関わる問題", "Safety", 5),
            ("清掃問題", "清掃・整理整頓の問題", "Maintenance", 2),
            ("人的要因", "作業者の行動に関わる問題", "Human", 3),
            ("環境要因", "温度・湿度・照明等の環境問題", "Environment", 2),
            ("その他", "上記に当てはまらない異常", "Other", 1)
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for type_name, description, category, severity in default_types:
                cursor.execute("""
                    INSERT OR IGNORE INTO anomaly_types (type_name, description, category, severity_level)
                    VALUES (?, ?, ?, ?)
                """, (type_name, description, category, severity))
            
            conn.commit()
    
    def start_feedback_session(self, user_id: str = "default") -> str:
        """
        フィードバックセッションを開始
        
        Args:
            user_id: ユーザーID
            
        Returns:
            セッションID
        """
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_sessions (session_id, user_id, start_time)
                VALUES (?, ?, ?)
            """, (session_id, user_id, datetime.now().isoformat()))
            
            conn.commit()
        
        return session_id
    
    def end_feedback_session(self, session_id: str):
        """フィードバックセッションを終了"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # セッションのフィードバック数を計算
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE session_id = ?", (session_id,))
            total_feedbacks = cursor.fetchone()[0]
            
            # セッション終了
            cursor.execute("""
                UPDATE feedback_sessions 
                SET end_time = ?, total_feedbacks = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), total_feedbacks, session_id))
            
            conn.commit()
    
    def add_feedback(self,
                    image_path: str,
                    is_anomaly: bool,
                    anomaly_type: str = None,
                    anomaly_description: str = "",
                    confidence_level: int = 3,
                    user_id: str = "default",
                    session_id: str = None,
                    image_id: int = None,
                    additional_metadata: Dict = None) -> str:
        """
        フィードバックを追加
        
        Args:
            image_path: 画像ファイルのパス
            is_anomaly: 異常かどうか
            anomaly_type: 異常タイプ
            anomaly_description: 異常の詳細説明
            confidence_level: 確信度（1-5）
            user_id: ユーザーID
            session_id: セッションID
            image_id: 画像ID
            additional_metadata: 追加メタデータ
            
        Returns:
            フィードバックID
        """
        feedback_id = str(uuid.uuid4())
        metadata_json = json.dumps(additional_metadata) if additional_metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback 
                (id, image_id, image_path, is_anomaly, anomaly_type, anomaly_description,
                 confidence_level, user_id, session_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (feedback_id, image_id, image_path, is_anomaly, anomaly_type, 
                  anomaly_description, confidence_level, user_id, session_id, metadata_json))
            
            conn.commit()
        
        return feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        """フィードバックを取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                feedback = dict(zip(columns, row))
                
                # JSON形式の追加メタデータをパース
                if feedback['metadata_json']:
                    try:
                        additional = json.loads(feedback['metadata_json'])
                        feedback.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                return feedback
            
            return None
    
    def list_feedbacks(self,
                      session_id: str = None,
                      user_id: str = None,
                      is_anomaly: bool = None,
                      anomaly_type: str = None,
                      limit: int = None) -> List[Dict]:
        """
        条件に基づいてフィードバックリストを取得
        
        Args:
            session_id: セッションID
            user_id: ユーザーID
            is_anomaly: 異常フラグ
            anomaly_type: 異常タイプ
            limit: 取得数制限
            
        Returns:
            フィードバック辞書のリスト
        """
        conditions = []
        params = []
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
            
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
            
        if is_anomaly is not None:
            conditions.append("is_anomaly = ?")
            params.append(is_anomaly)
            
        if anomaly_type:
            conditions.append("anomaly_type = ?")
            params.append(anomaly_type)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f"SELECT * FROM feedback{where_clause} ORDER BY created_at DESC{limit_clause}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                feedback = dict(zip(columns, row))
                
                # JSON形式の追加メタデータをパース
                if feedback['metadata_json']:
                    try:
                        additional = json.loads(feedback['metadata_json'])
                        feedback.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                results.append(feedback)
            
            return results
    
    def get_anomaly_types(self) -> List[Dict]:
        """異常タイプ一覧を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM anomaly_types ORDER BY severity_level DESC, type_name")
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def get_feedback_statistics(self) -> Dict:
        """フィードバック統計情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 総フィードバック数
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedbacks = cursor.fetchone()[0]
            
            # 異常・正常の比率
            cursor.execute("SELECT is_anomaly, COUNT(*) FROM feedback GROUP BY is_anomaly")
            anomaly_counts = dict(cursor.fetchall())
            
            # 異常タイプ別統計
            cursor.execute("""
                SELECT anomaly_type, COUNT(*) 
                FROM feedback 
                WHERE is_anomaly = 1 AND anomaly_type IS NOT NULL
                GROUP BY anomaly_type 
                ORDER BY COUNT(*) DESC
            """)
            anomaly_type_stats = dict(cursor.fetchall())
            
            # セッション別統計
            cursor.execute("SELECT COUNT(*) FROM feedback_sessions")
            total_sessions = cursor.fetchone()[0]
            
            # 確信度別統計
            cursor.execute("SELECT confidence_level, COUNT(*) FROM feedback GROUP BY confidence_level")
            confidence_stats = dict(cursor.fetchall())
            
            return {
                "total_feedbacks": total_feedbacks,
                "total_sessions": total_sessions,
                "anomaly_normal_ratio": anomaly_counts,
                "anomaly_type_statistics": anomaly_type_stats,
                "confidence_level_statistics": confidence_stats
            }
    
    def export_feedbacks_to_json(self, output_path: str, session_id: str = None):
        """フィードバックをJSON形式でエクスポート"""
        feedbacks = self.list_feedbacks(session_id=session_id)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "total_feedbacks": len(feedbacks),
            "feedbacks": feedbacks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return export_data
    
    def create_training_dataset(self, output_dir: str, session_id: str = None) -> Dict:
        """
        学習用データセットを作成
        
        Args:
            output_dir: 出力ディレクトリ
            session_id: 特定のセッションのみを使用する場合
            
        Returns:
            データセット情報
        """
        feedbacks = self.list_feedbacks(session_id=session_id)
        
        # フィードバック付き画像とテキストペアの作成
        training_data = []
        
        for feedback in feedbacks:
            if feedback['is_anomaly'] and feedback['anomaly_description']:
                data_entry = {
                    "image_path": feedback['image_path'],
                    "anomaly_type": feedback['anomaly_type'],
                    "anomaly_description": feedback['anomaly_description'],
                    "confidence_level": feedback['confidence_level'],
                    "feedback_id": feedback['id']
                }
                training_data.append(data_entry)
        
        # 出力ディレクトリの作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSONL形式で保存（機械学習用）
        jsonl_path = output_path / "training_data.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in training_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 統計情報も保存
        dataset_info = {
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
            "total_samples": len(training_data),
            "anomaly_type_counts": {},
            "jsonl_path": str(jsonl_path)
        }
        
        # 異常タイプ別カウント
        for entry in training_data:
            anomaly_type = entry['anomaly_type']
            dataset_info["anomaly_type_counts"][anomaly_type] = \
                dataset_info["anomaly_type_counts"].get(anomaly_type, 0) + 1
        
        # 情報ファイル保存
        info_path = output_path / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        return dataset_info


if __name__ == "__main__":
    # テスト用のサンプル実行
    import tempfile
    
    # 一時的なデータベースで動作確認
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    manager = FeedbackDataManager(db_path)
    
    # セッション開始
    session_id = manager.start_feedback_session("test_user")
    print(f"Started session: {session_id}")
    
    # サンプルフィードバック追加
    feedback_id = manager.add_feedback(
        image_path="test_image.jpg",
        is_anomaly=True,
        anomaly_type="設備故障",
        anomaly_description="機械の部品が破損している",
        confidence_level=4,
        session_id=session_id
    )
    
    print(f"Added feedback: {feedback_id}")
    
    # 統計情報表示
    stats = manager.get_feedback_statistics()
    print("Statistics:", stats)
    
    # セッション終了
    manager.end_feedback_session(session_id)
    
    # クリーンアップ
    os.unlink(db_path)
