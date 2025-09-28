"""
MAD-FH: Image Metadata Database Manager
画像のメタデータ（撮影日時、カメラID、設置場所など）をSQLiteで管理
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ImageMetadataDB:
    """画像メタデータの管理クラス"""
    
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    camera_id TEXT,
                    location TEXT,
                    capture_datetime TEXT,
                    width INTEGER,
                    height INTEGER,
                    file_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_processed BOOLEAN DEFAULT FALSE,
                    metadata_json TEXT
                )
            """)
            
            # インデックス作成
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_id ON images(camera_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_capture_datetime ON images(capture_datetime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_processed ON images(is_processed)")
            
            conn.commit()
    
    def add_image(self, 
                  image_path: str,
                  camera_id: str = None,
                  location: str = None,
                  capture_datetime: str = None,
                  width: int = None,
                  height: int = None,
                  file_size: int = None,
                  additional_metadata: Dict = None) -> int:
        """
        画像メタデータを追加
        
        Args:
            image_path: 画像ファイルのパス
            camera_id: カメラID
            location: 設置場所
            capture_datetime: 撮影日時（ISO形式文字列）
            width: 画像幅
            height: 画像高さ
            file_size: ファイルサイズ（バイト）
            additional_metadata: 追加メタデータ（JSON形式で保存）
            
        Returns:
            追加されたレコードのID
        """
        filename = Path(image_path).name
        
        # ファイル情報を自動取得
        if os.path.exists(image_path):
            stat = os.stat(image_path)
            if file_size is None:
                file_size = stat.st_size
                
        # 撮影日時をファイル名から推定（YYYYMMDD_HHMMSS形式の場合）
        if capture_datetime is None:
            try:
                # ファイル名から日時を抽出
                datetime_part = filename.split('_')[:2]  # YYYYMMDD_HHMMSS
                if len(datetime_part) == 2:
                    date_str, time_str = datetime_part
                    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    capture_datetime = dt.isoformat()
            except ValueError:
                # パースできない場合はファイルの更新時刻を使用
                if os.path.exists(image_path):
                    mtime = os.path.getmtime(image_path)
                    capture_datetime = datetime.fromtimestamp(mtime).isoformat()
        
        metadata_json = json.dumps(additional_metadata) if additional_metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO images 
                (image_path, filename, camera_id, location, capture_datetime,
                 width, height, file_size, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_path, filename, camera_id, location, capture_datetime,
                  width, height, file_size, metadata_json))
            
            return cursor.lastrowid
    
    def get_image_metadata(self, image_id: int = None, image_path: str = None) -> Optional[Dict]:
        """
        画像メタデータを取得
        
        Args:
            image_id: 画像ID
            image_path: 画像パス
            
        Returns:
            メタデータ辞書、見つからない場合はNone
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if image_id:
                cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            elif image_path:
                cursor.execute("SELECT * FROM images WHERE image_path = ?", (image_path,))
            else:
                return None
                
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                metadata = dict(zip(columns, row))
                
                # JSON形式の追加メタデータをパース
                if metadata['metadata_json']:
                    try:
                        additional = json.loads(metadata['metadata_json'])
                        metadata.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                return metadata
            
            return None
    
    def list_images(self, 
                   camera_id: str = None,
                   location: str = None,
                   date_from: str = None,
                   date_to: str = None,
                   is_processed: bool = None) -> List[Dict]:
        """
        条件に基づいて画像リストを取得
        
        Args:
            camera_id: カメラID
            location: 設置場所
            date_from: 開始日時（ISO形式）
            date_to: 終了日時（ISO形式）
            is_processed: 処理済みフラグ
            
        Returns:
            メタデータ辞書のリスト
        """
        conditions = []
        params = []
        
        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
            
        if location:
            conditions.append("location = ?")
            params.append(location)
            
        if date_from:
            conditions.append("capture_datetime >= ?")
            params.append(date_from)
            
        if date_to:
            conditions.append("capture_datetime <= ?")
            params.append(date_to)
            
        if is_processed is not None:
            conditions.append("is_processed = ?")
            params.append(is_processed)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM images{where_clause} ORDER BY capture_datetime"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                metadata = dict(zip(columns, row))
                
                # JSON形式の追加メタデータをパース
                if metadata['metadata_json']:
                    try:
                        additional = json.loads(metadata['metadata_json'])
                        metadata.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                results.append(metadata)
            
            return results
    
    def mark_processed(self, image_id: int, processed: bool = True):
        """画像の処理済みフラグを更新"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE images SET is_processed = ? WHERE id = ?", 
                          (processed, image_id))
    
    def get_statistics(self) -> Dict:
        """データベースの統計情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 総画像数
            cursor.execute("SELECT COUNT(*) FROM images")
            total_images = cursor.fetchone()[0]
            
            # 処理済み画像数
            cursor.execute("SELECT COUNT(*) FROM images WHERE is_processed = TRUE")
            processed_images = cursor.fetchone()[0]
            
            # カメラ別統計
            cursor.execute("SELECT camera_id, COUNT(*) FROM images GROUP BY camera_id")
            camera_stats = dict(cursor.fetchall())
            
            # 日付範囲
            cursor.execute("SELECT MIN(capture_datetime), MAX(capture_datetime) FROM images")
            date_range = cursor.fetchone()
            
            return {
                "total_images": total_images,
                "processed_images": processed_images,
                "unprocessed_images": total_images - processed_images,
                "camera_statistics": camera_stats,
                "date_range": {
                    "start": date_range[0],
                    "end": date_range[1]
                }
            }


def scan_and_register_images(image_directory: str, 
                           db_path: str,
                           camera_id: str = None,
                           location: str = None) -> int:
    """
    ディレクトリをスキャンして画像をデータベースに登録
    
    Args:
        image_directory: 画像ディレクトリパス
        db_path: データベースパス
        camera_id: デフォルトカメラID
        location: デフォルト設置場所
        
    Returns:
        登録された画像数
    """
    db = ImageMetadataDB(db_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_dir = Path(image_directory)
    registered_count = 0
    
    for image_path in image_dir.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            try:
                # 画像サイズを取得
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # データベースに登録
                db.add_image(
                    image_path=str(image_path),
                    camera_id=camera_id,
                    location=location,
                    width=width,
                    height=height
                )
                registered_count += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    return registered_count


if __name__ == "__main__":
    # テスト用のサンプル実行
    import tempfile
    
    # 一時的なデータベースで動作確認
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    db = ImageMetadataDB(db_path)
    
    # サンプルデータ追加
    image_id = db.add_image(
        image_path="data/images/normal/20240911_120000_cam001.jpg",
        camera_id="cam001",
        location="factory_entrance",
        width=1920,
        height=1080,
        additional_metadata={"quality": "high", "lighting": "natural"}
    )
    
    print(f"Added image with ID: {image_id}")
    
    # 統計情報表示
    stats = db.get_statistics()
    print("Database Statistics:", stats)
    
    # クリーンアップ
    os.unlink(db_path)
