"""
ナレッジベース機能のテスト用スクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.knowledge_base.knowledge_manager import KnowledgeBaseManager
from src.knowledge_base.vector_database import FeatureRecord
import yaml
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """テスト用データを作成"""
    try:
        # 設定読み込み
        config_path = project_root / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # ナレッジベースマネージャー初期化
        kb_manager = KnowledgeBaseManager(config)
        
        # テストデータ
        test_features = [
            {
                "description": "製品表面に小さな黒い点状の異常が検出されました。直径約2mm、深度は軽微です。",
                "category": "anomaly",
                "confidence": 0.95,
                "source": "test_data",
                "metadata": {
                    "location": "ライン1",
                    "equipment": "検査装置A",
                    "severity": "low",
                    "tags": ["表面異常", "点状", "黒色"]
                }
            },
            {
                "description": "エッジ部分に微細なクラックが発生。長さ約5mm、幅0.1mm程度の線状欠陥です。",
                "category": "anomaly", 
                "confidence": 0.88,
                "source": "test_data",
                "metadata": {
                    "location": "ライン2",
                    "equipment": "検査装置B", 
                    "severity": "medium",
                    "tags": ["エッジ異常", "クラック", "線状"]
                }
            },
            {
                "description": "正常な製品表面状態。滑らかで均一な仕上がりが確認されています。",
                "category": "normal",
                "confidence": 0.98,
                "source": "test_data",
                "metadata": {
                    "location": "ライン1",
                    "equipment": "検査装置A",
                    "severity": "none",
                    "tags": ["正常", "滑らか", "均一"]
                }
            },
            {
                "description": "色むらが軽度に発生。全体的な品質は許容範囲内ですが注意が必要です。",
                "category": "anomaly",
                "confidence": 0.72,
                "source": "test_data", 
                "metadata": {
                    "location": "ライン3",
                    "equipment": "検査装置C",
                    "severity": "low",
                    "tags": ["色むら", "軽度", "注意"]
                }
            },
            {
                "description": "深刻な変形が検出されました。製品の形状が大きく歪んでいます。",
                "category": "anomaly",
                "confidence": 0.99,
                "source": "test_data",
                "metadata": {
                    "location": "ライン2", 
                    "equipment": "検査装置B",
                    "severity": "critical",
                    "tags": ["変形", "歪み", "深刻"]
                }
            },
            {
                "description": "標準的な品質基準を満たす製品。寸法、表面状態ともに良好です。",
                "category": "normal",
                "confidence": 0.96,
                "source": "test_data",
                "metadata": {
                    "location": "ライン1",
                    "equipment": "検査装置A",
                    "severity": "none", 
                    "tags": ["標準", "良好", "寸法適正"]
                }
            }
        ]
        
        # テストデータを追加
        added_count = 0
        for feature_data in test_features:
            feature_id = kb_manager.add_feature(**feature_data)
            if feature_id:
                added_count += 1
                logger.info(f"テストデータ追加: {feature_id}")
        
        logger.info(f"テストデータ追加完了: {added_count}件")
        
        # 統計情報確認
        stats = kb_manager.get_knowledge_base_stats()
        logger.info(f"ナレッジベース統計: {stats}")
        
        # 検索テスト
        search_results = kb_manager.search_similar_features("黒い点", similarity_threshold=0.5)
        logger.info(f"検索結果: {len(search_results)}件")
        for result in search_results:
            logger.info(f"  - {result['description'][:50]}... (類似度: {result.get('similarity', 0):.3f})")
        
        # クリーンアップ
        kb_manager.close()
        
        return True
        
    except Exception as e:
        logger.error(f"テストデータ作成エラー: {e}")
        return False

if __name__ == "__main__":
    print("ナレッジベーステストデータを作成しています...")
    success = create_test_data()
    
    if success:
        print("✅ テストデータ作成完了！")
        print("   Streamlit UIでナレッジベースページをテストできます。")
    else:
        print("❌ テストデータ作成に失敗しました。")
