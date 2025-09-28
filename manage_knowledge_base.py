#!/usr/bin/env python3
"""
Knowledge Base Management Script
ナレッジベースの管理・更新・分析用スクリプト

使用方法:
    python manage_knowledge_base.py --update                    # ナレッジベース更新
    python manage_knowledge_base.py --search "異常説明"         # 類似事例検索
    python manage_knowledge_base.py --stats                     # 統計情報表示
    python manage_knowledge_base.py --export summary.json       # サマリー出力
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.knowledge_base.knowledge_manager import KnowledgeBaseManager

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/knowledge_base.log', encoding='utf-8')
        ]
    )

def load_config(config_path: str):
    """設定ファイル読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def update_knowledge_base(kb_manager: KnowledgeBaseManager):
    """ナレッジベース更新"""
    print("🔄 ナレッジベースを更新中...")
    
    result = kb_manager.update_knowledge_base(force_update=True)
    
    print(f"✅ 更新完了!")
    print(f"  - 新規特徴量: {result.get('extracted_features', 0)}件")
    print(f"  - 総レコード数: {result.get('total_records', 0)}件")
    print(f"  - ベクトルDB: {result.get('vector_db_size', 0)}件")

def search_similar_cases(kb_manager: KnowledgeBaseManager, query: str):
    """類似事例検索"""
    print(f"🔍 類似事例検索: '{query}'")
    
    results = kb_manager.find_similar_cases(query, top_k=5)
    
    if not results:
        print("❌ 類似事例が見つかりませんでした")
        return
    
    print(f"✅ {len(results)}件の類似事例が見つかりました:\n")
    
    for i, result in enumerate(results, 1):
        print(f"【{i}】 {result['anomaly_type']}")
        print(f"    説明: {result['anomaly_description']}")
        print(f"    類似度: {result['similarity_score']:.3f}")
        print(f"    信頼度: {result['confidence_score']}")
        print(f"    画像: {result['image_path']}")
        print()

def show_statistics(kb_manager: KnowledgeBaseManager):
    """統計情報表示"""
    print("📊 ナレッジベース統計情報")
    
    stats = kb_manager.knowledge_base.get_statistics()
    
    print(f"総レコード数: {stats.get('total_records', 0)}件")
    print(f"最新更新: {stats.get('latest_timestamp', 'N/A')}")
    print()
    
    anomaly_types = stats.get('anomaly_types', {})
    if anomaly_types:
        print("異常タイプ別統計:")
        for anomaly_type, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {anomaly_type}: {count}件")
    else:
        print("異常タイプデータがありません")
    
    print()
    
    # 詳細分析
    for anomaly_type in list(anomaly_types.keys())[:3]:  # 上位3タイプ
        patterns = kb_manager.get_anomaly_patterns(anomaly_type)
        type_analysis = patterns.get('type_analysis', {})
        
        if type_analysis:
            print(f"【{anomaly_type}】詳細分析:")
            print(f"  - 平均信頼度: {type_analysis.get('avg_confidence', 0):.2f}")
            print(f"  - 信頼度範囲: {type_analysis.get('min_confidence', 0):.1f} - {type_analysis.get('max_confidence', 0):.1f}")
            print()

def export_summary(kb_manager: KnowledgeBaseManager, output_path: str):
    """サマリー出力"""
    print(f"📄 サマリーを出力中: {output_path}")
    
    success = kb_manager.export_knowledge_summary(output_path)
    
    if success:
        print(f"✅ サマリー出力完了: {output_path}")
    else:
        print("❌ サマリー出力に失敗しました")

def suggest_description(kb_manager: KnowledgeBaseManager, anomaly_type: str):
    """異常説明提案"""
    print(f"💡 異常説明提案: '{anomaly_type}'")
    
    suggestion = kb_manager.suggest_anomaly_description(
        image_path="",
        detected_anomaly_type=anomaly_type
    )
    
    if not suggestion:
        print("❌ 提案の生成に失敗しました")
        return
    
    print(f"✅ 提案された説明:")
    print(f"  メイン: {suggestion.get('suggested_description', 'N/A')}")
    print(f"  信頼度: {suggestion.get('confidence', 0):.3f}")
    print(f"  類似事例数: {suggestion.get('similar_cases_count', 0)}件")
    
    alternatives = suggestion.get('alternative_descriptions', [])
    if alternatives:
        print(f"  代替案:")
        for i, alt in enumerate(alternatives[:3], 1):
            print(f"    {i}. {alt}")
    
    common_phrases = suggestion.get('common_phrases', [])
    if common_phrases:
        print(f"  共通キーワード: {', '.join(common_phrases[:5])}")

def main():
    parser = argparse.ArgumentParser(description='ナレッジベース管理ツール')
    parser.add_argument('--config', default='config/config.yaml', help='設定ファイルパス')
    parser.add_argument('--update', action='store_true', help='ナレッジベース更新')
    parser.add_argument('--search', type=str, help='類似事例検索')
    parser.add_argument('--stats', action='store_true', help='統計情報表示')
    parser.add_argument('--export', type=str, help='サマリー出力ファイルパス')
    parser.add_argument('--suggest', type=str, help='異常説明提案（異常タイプ指定）')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    
    try:
        # 設定読み込み
        config = load_config(args.config)
        
        # ナレッジベースマネージャー初期化
        kb_manager = KnowledgeBaseManager(config)
        
        print("🚀 MAD-FH Knowledge Base Manager")
        print("=" * 50)
        
        # コマンド実行
        if args.update:
            update_knowledge_base(kb_manager)
        elif args.search:
            search_similar_cases(kb_manager, args.search)
        elif args.stats:
            show_statistics(kb_manager)
        elif args.export:
            export_summary(kb_manager, args.export)
        elif args.suggest:
            suggest_description(kb_manager, args.suggest)
        else:
            print("コマンドを指定してください。--help でヘルプを表示します。")
        
        # クリーンアップ
        kb_manager.close()
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
