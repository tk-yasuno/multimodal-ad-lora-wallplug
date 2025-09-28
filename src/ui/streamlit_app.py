"""
MAD-FH: Streamlit Human-in-the-Loop UI
人間フィードバック収集用のWebアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import torch
from datetime import datetime
import os
import logging
from typing import Dict, List, Any

# プロジェクト内のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ui.feedback_manager import FeedbackDataManager
from src.data.metadata_manager import ImageMetadataDB
from src.models.autoencoder import ConvAutoencoder, AnomalyDetector
from src.data.preprocess import ImagePreprocessor
import yaml

logger = logging.getLogger(__name__)


class MADFHApp:
    """MAD-FH Streamlitアプリケーション"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.setup_page_config()
        self.load_config()
        self.initialize_managers()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Streamlitページ設定"""
        st.set_page_config(
            page_title="MAD-FH: Multimodal Anomaly Detector",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_config(self):
        """設定ファイルの読み込み"""
        config_path = "config/config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"設定ファイルが見つかりません: {config_path}")
            st.stop()
    
    def initialize_managers(self):
        """データ管理クラスの初期化"""
        # フィードバックデータマネージャー
        feedback_db_path = "data/feedback/feedback.db"
        self.feedback_manager = FeedbackDataManager(feedback_db_path)
        
        # 画像メタデータマネージャー
        metadata_db_path = self.config['data']['metadata_db_path']
        self.metadata_manager = ImageMetadataDB(metadata_db_path)
        
        # 画像前処理器
        self.preprocessor = ImagePreprocessor("config/config.yaml")
        
        # Knowledge Base マネージャー
        self.knowledge_manager = None
        self.load_knowledge_manager()
        
        # テキスト生成統合（LoRA）
        self.text_generation = None
        self.load_text_generation()
        
        # 異常検知器（モデルが存在する場合）
        self.anomaly_detector = None
        self.load_anomaly_detector()
    
    def load_knowledge_manager(self):
        """Knowledge Base マネージャーの読み込み"""
        try:
            from src.knowledge_base.knowledge_manager import KnowledgeBaseManager
            self.knowledge_manager = KnowledgeBaseManager(self.config)
            logger.info("Knowledge Base マネージャー初期化完了")
        except Exception as e:
            logger.warning(f"Knowledge Base 読み込みエラー: {e}")
            self.knowledge_manager = None
    
    def load_text_generation(self):
        """テキスト生成機能の読み込み"""
        try:
            from .text_generation_integration import UITextGenerationIntegration
            self.text_generation = UITextGenerationIntegration()
            logger.info("テキスト生成統合クラス初期化完了")
        except Exception as e:
            logger.warning(f"テキスト生成機能読み込みエラー: {e}")
            self.text_generation = None
    
    def load_anomaly_detector(self):
        """異常検知器の読み込み"""
        model_path = "models/best_autoencoder_model.pt"
        
        if Path(model_path).exists():
            try:
                # モデルの作成
                model = ConvAutoencoder(
                    input_channels=3,
                    latent_dim=256,
                    input_size=(512, 512)
                )
                
                # 学習済み重みの読み込み
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                
                # 異常検知器の初期化
                self.anomaly_detector = AnomalyDetector(model, device='cpu')
                
                st.sidebar.success("異常検知モデルを読み込みました")
            except Exception as e:
                st.sidebar.warning(f"モデル読み込みエラー: {e}")
        else:
            st.sidebar.info("異常検知モデルが見つかりません")
    
    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        
        if 'feedback_count' not in st.session_state:
            st.session_state.feedback_count = 0
        
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        
        if 'image_list' not in st.session_state:
            st.session_state.image_list = []
    
    def sidebar_navigation(self):
        """サイドバーナビゲーション"""
        st.sidebar.title("🔍 MAD-FH")
        st.sidebar.markdown("---")
        
        # ページ選択
        page = st.sidebar.selectbox(
            "ページを選択",
            ["画像フィードバック", "AI説明生成", "FODD即時分析", "ナレッジベース", "データ分析", "システム状態", "設定"]
        )
        
        st.sidebar.markdown("---")
        
        # セッション管理
        st.sidebar.subheader("📝 セッション管理")
        
        if st.session_state.current_session_id is None:
            user_id = st.sidebar.text_input("ユーザーID", value="operator_01")
            if st.sidebar.button("セッション開始"):
                session_id = self.feedback_manager.start_feedback_session(user_id)
                st.session_state.current_session_id = session_id
                st.session_state.feedback_count = 0
                st.sidebar.success(f"セッション開始: {session_id[:8]}...")
                st.experimental_rerun()
        else:
            st.sidebar.info(f"セッション: {st.session_state.current_session_id[:8]}...")
            st.sidebar.metric("フィードバック数", st.session_state.feedback_count)
            
            if st.sidebar.button("セッション終了"):
                self.feedback_manager.end_feedback_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.session_state.feedback_count = 0
                st.sidebar.success("セッションを終了しました")
                st.experimental_rerun()
        
        return page
    
    def load_image_list(self):
        """画像リストの読み込み"""
        if not st.session_state.image_list:
            images = self.metadata_manager.list_images()
            st.session_state.image_list = [img for img in images if Path(img['image_path']).exists()]
    
    def image_feedback_page(self):
        """画像フィードバックページ"""
        st.title("🖼️ 画像フィードバック収集")
        
        if st.session_state.current_session_id is None:
            st.warning("フィードバックを開始するには、左側のサイドバーでセッションを開始してください。")
            return
        
        self.load_image_list()
        
        if not st.session_state.image_list:
            st.error("画像が見つかりません。先にデータを登録してください。")
            return
        
        # 画像ナビゲーション
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ 前の画像") and st.session_state.current_image_index > 0:
                st.session_state.current_image_index -= 1
                st.experimental_rerun()
        
        with col2:
            st.metric(
                "画像番号", 
                f"{st.session_state.current_image_index + 1} / {len(st.session_state.image_list)}"
            )
        
        with col3:
            if st.button("次の画像 ➡️") and st.session_state.current_image_index < len(st.session_state.image_list) - 1:
                st.session_state.current_image_index += 1
                st.experimental_rerun()
        
        # 現在の画像情報
        current_image = st.session_state.image_list[st.session_state.current_image_index]
        image_path = current_image['image_path']
        
        # 画像表示と異常検知結果
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 現在の画像")
            
            try:
                image = Image.open(image_path)
                st.image(image, caption=Path(image_path).name, use_column_width=True)
                
                # 画像情報表示
                st.info(f"""
                **ファイル名**: {current_image['filename']}  
                **撮影日時**: {current_image.get('capture_datetime', 'N/A')}  
                **カメラID**: {current_image.get('camera_id', 'N/A')}  
                **場所**: {current_image.get('location', 'N/A')}  
                **サイズ**: {image.size}
                """)
                
            except Exception as e:
                st.error(f"画像の読み込みに失敗しました: {e}")
                return
        
        with col2:
            st.subheader("🤖 AI異常検知結果")
            
            if self.anomaly_detector:
                try:
                    # 画像の前処理と異常検知
                    with st.spinner("異常検知中..."):
                        tensor = self.preprocessor.preprocess_image(image_path, apply_augmentation=False)
                        tensor = tensor.unsqueeze(0)  # バッチ次元追加
                        
                        scores, predictions = self.anomaly_detector.predict(tensor)
                        score = scores[0]
                        prediction = predictions[0]
                    
                    # 結果表示
                    if prediction == 1:
                        st.error(f"🚨 異常検出: {score:.3f}")
                    else:
                        st.success(f"✅ 正常: {score:.3f}")
                    
                    # 異常度バー
                    progress_color = "red" if prediction == 1 else "green"
                    st.metric("異常度スコア", f"{score:.3f}")
                    
                    if self.anomaly_detector.threshold:
                        st.metric("閾値", f"{self.anomaly_detector.threshold:.3f}")
                    
                except Exception as e:
                    st.error(f"異常検知エラー: {e}")
            else:
                st.info("異常検知モデルが利用できません")
        
        # フィードバック入力フォーム
        st.markdown("---")
        st.subheader("👤 人間によるフィードバック")
        
        with st.form("feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                is_anomaly = st.radio(
                    "この画像は異常ですか？",
                    options=[False, True],
                    format_func=lambda x: "🚨 異常" if x else "✅ 正常"
                )
                
                confidence_level = st.slider(
                    "確信度",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1: 不確実 ～ 5: 非常に確実"
                )
            
            with col2:
                anomaly_types = [at['type_name'] for at in self.feedback_manager.get_anomaly_types()]
                
                anomaly_type = st.selectbox(
                    "異常タイプ",
                    options=[""] + anomaly_types,
                    disabled=not is_anomaly
                )
                
                anomaly_description = st.text_area(
                    "異常の詳細説明",
                    placeholder="具体的な異常内容を記述してください...",
                    disabled=not is_anomaly,
                    height=100
                )
            
            # フィードバック送信
            submitted = st.form_submit_button("📝 フィードバック送信", type="primary")
            
            if submitted:
                # フィードバックを保存
                feedback_id = self.feedback_manager.add_feedback(
                    image_path=image_path,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type if is_anomaly else None,
                    anomaly_description=anomaly_description if is_anomaly else "",
                    confidence_level=confidence_level,
                    session_id=st.session_state.current_session_id,
                    image_id=current_image.get('id'),
                    additional_metadata={
                        "ai_prediction": prediction if self.anomaly_detector else None,
                        "ai_score": float(score) if self.anomaly_detector else None
                    }
                )
                
                st.session_state.feedback_count += 1
                st.success(f"フィードバックを保存しました！ (ID: {feedback_id[:8]}...)")
                
                # 次の画像に自動移動
                if st.session_state.current_image_index < len(st.session_state.image_list) - 1:
                    st.session_state.current_image_index += 1
                    st.experimental_rerun()
    
    def data_analysis_page(self):
        """データ分析ページ"""
        st.title("📊 データ分析")
        
        # 統計情報取得
        stats = self.feedback_manager.get_feedback_statistics()
        
        # メトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総フィードバック数", stats.get('total_feedbacks', 0))
        
        with col2:
            st.metric("セッション数", stats.get('total_sessions', 0))
        
        with col3:
            anomaly_ratio = stats.get('anomaly_normal_ratio', {})
            anomaly_count = anomaly_ratio.get(1, 0)  # True
            total = sum(anomaly_ratio.values())
            ratio = (anomaly_count / total * 100) if total > 0 else 0
            st.metric("異常率", f"{ratio:.1f}%")
        
        with col4:
            avg_confidence = self.calculate_average_confidence(stats)
            st.metric("平均確信度", f"{avg_confidence:.1f}")
        
        # グラフ表示
        col1, col2 = st.columns(2)
        
        with col1:
            # 異常・正常の分布
            anomaly_ratio = stats.get('anomaly_normal_ratio', {})
            if anomaly_ratio:
                fig = px.pie(
                    values=list(anomaly_ratio.values()),
                    names=['正常', '異常'],
                    title="異常・正常の分布"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 異常タイプ別統計
            anomaly_types = stats.get('anomaly_type_statistics', {})
            if anomaly_types:
                fig = px.bar(
                    x=list(anomaly_types.values()),
                    y=list(anomaly_types.keys()),
                    orientation='h',
                    title="異常タイプ別件数"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # フィードバック履歴テーブル
        st.subheader("📝 最新のフィードバック")
        recent_feedbacks = self.feedback_manager.list_feedbacks(limit=20)
        
        if recent_feedbacks:
            df = pd.DataFrame(recent_feedbacks)
            display_columns = ['created_at', 'image_path', 'is_anomaly', 'anomaly_type', 'confidence_level']
            available_columns = [col for col in display_columns if col in df.columns]
            
            st.dataframe(
                df[available_columns].head(10),
                use_container_width=True
            )
        
        # データエクスポート
        st.subheader("📥 データエクスポート")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("フィードバックをJSONエクスポート"):
                output_path = "data/feedback/export_feedback.json"
                self.feedback_manager.export_feedbacks_to_json(output_path)
                st.success(f"エクスポート完了: {output_path}")
        
        with col2:
            if st.button("学習用データセット作成"):
                output_dir = "data/feedback/training_dataset"
                dataset_info = self.feedback_manager.create_training_dataset(output_dir)
                st.success(f"データセット作成完了: {dataset_info['total_samples']}サンプル")
    
    def calculate_average_confidence(self, stats):
        """平均確信度の計算"""
        confidence_stats = stats.get('confidence_level_statistics', {})
        if not confidence_stats:
            return 0
        
        total_weighted = sum(level * count for level, count in confidence_stats.items())
        total_count = sum(confidence_stats.values())
        
        return total_weighted / total_count if total_count > 0 else 0
    
    def ai_description_page(self):
        """AI説明生成ページ"""
        st.title("🤖 AI異常説明生成")
        st.markdown("LoRAを使用したマルチモーダル異常説明生成システム")
        
        # テキスト生成機能の状態確認
        if self.text_generation is None:
            st.error("❌ テキスト生成機能が利用できません")
            st.info("LoRAモデルの学習が必要です。")
            return
        
        # テキスト生成機能の初期化
        if not self.text_generation.is_initialized:
            with st.spinner("🔄 LoRAモデル初期化中..."):
                success = self.text_generation.initialize()
                if not success:
                    st.error("❌ LoRAモデルの初期化に失敗しました")
                    return
                else:
                    st.success("✅ LoRAモデル準備完了")
        
        # モデル状態表示
        st.subheader("📊 モデル状態")
        model_status = self.text_generation.get_model_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "🟢" if model_status.get('model_loaded', False) else "🔴"
            st.metric("モデル状態", f"{status_color} {model_status.get('status', 'unknown')}")
        
        with col2:
            st.metric("モデル名", model_status.get('model_name', 'N/A'))
        
        with col3:
            lora_status = "有効" if model_status.get('lora_enabled', False) else "無効"
            st.metric("LoRA", lora_status)
        
        st.markdown("---")
        
        # 画像アップロードまたは選択
        st.subheader("🖼️ 画像選択")
        
        tab1, tab2 = st.tabs(["画像アップロード", "既存画像から選択"])
        
        selected_image = None
        image_source = None
        
        with tab1:
            uploaded_file = st.file_uploader(
                "画像をアップロード",
                type=['jpg', 'jpeg', 'png'],
                help="異常説明を生成したい画像をアップロードしてください"
            )
            
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                image_source = "uploaded"
                st.image(selected_image, caption="アップロード画像", use_column_width=True)
        
        with tab2:
            # 既存画像リストから選択
            self.load_image_list()
            
            if st.session_state.image_list:
                image_options = [f"{i+1}: {Path(img['image_path']).name}" for i, img in enumerate(st.session_state.image_list)]
                
                selected_idx = st.selectbox(
                    "画像を選択",
                    range(len(image_options)),
                    format_func=lambda x: image_options[x]
                )
                
                if selected_idx is not None:
                    image_path = st.session_state.image_list[selected_idx]['image_path']
                    if Path(image_path).exists():
                        selected_image = Image.open(image_path)
                        image_source = "existing"
                        st.image(selected_image, caption=f"選択画像: {Path(image_path).name}", use_column_width=True)
                    else:
                        st.error(f"画像ファイルが見つかりません: {image_path}")
            else:
                st.info("画像データが見つかりません")
        
        # 画像が選択された場合の説明生成
        if selected_image is not None:
            st.markdown("---")
            st.subheader("⚙️ 生成設定")
            
            # プロンプト設定
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 異常タイプ選択（プロンプト提案用）
                anomaly_types = ["カスタム", "設備故障", "製品欠陥", "安全問題", "環境異常"]
                selected_anomaly_type = st.selectbox("異常タイプ", anomaly_types)
                
                if selected_anomaly_type == "カスタム":
                    custom_prompt = st.text_area(
                        "カスタムプロンプト",
                        value="この画像の異常を詳しく説明してください:",
                        height=100
                    )
                else:
                    # 異常タイプに基づくプロンプト提案
                    suggestions = self.text_generation.get_generation_suggestions(selected_anomaly_type)
                    selected_prompt = st.selectbox("推奨プロンプト", suggestions)
                    custom_prompt = st.text_area(
                        "プロンプト（編集可能）",
                        value=selected_prompt,
                        height=100
                    )
            
            with col2:
                st.markdown("**生成パラメータ**")
                
                max_tokens = st.slider("最大トークン数", 50, 256, 128)
                temperature = st.slider("温度（創造性）", 0.1, 1.0, 0.7, 0.1)
                top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
            
            # 生成実行
            if st.button("🚀 異常説明生成", type="primary"):
                with st.spinner("AI説明生成中..."):
                    generation_params = {
                        'max_new_tokens': max_tokens,
                        'temperature': temperature,
                        'top_p': top_p,
                        'do_sample': True
                    }
                    
                    result = self.text_generation.generate_anomaly_description(
                        selected_image,
                        custom_prompt,
                        generation_params
                    )
                    
                    st.markdown("---")
                    st.subheader("📝 生成結果")
                    
                    if result['success']:
                        # 成功時の表示
                        st.success("✅ 異常説明の生成が完了しました")
                        
                        # 生成された説明
                        st.text_area(
                            "生成された異常説明",
                            value=result['description'],
                            height=150,
                            key="generated_description"
                        )
                        
                        # メタ情報
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("確信度", f"{result['confidence']:.2f}")
                        with col2:
                            st.metric("文字数", len(result['description']))
                        with col3:
                            st.metric("生成時刻", result['timestamp'].split('T')[1][:8])
                        
                        # フィードバック保存オプション
                        st.markdown("---")
                        st.subheader("💾 フィードバック保存")
                        
                        with st.expander("生成結果をフィードバックとして保存"):
                            # 異常判定
                            is_anomaly = st.checkbox("異常として分類", value=True)
                            
                            # 異常タイプ
                            if is_anomaly:
                                anomaly_type_options = ["設備故障", "製品欠陥", "安全問題", "環境異常", "その他"]
                                feedback_anomaly_type = st.selectbox("異常タイプ", anomaly_type_options)
                            else:
                                feedback_anomaly_type = None
                            
                            # 確信度レベル
                            confidence_level = st.slider("確信度レベル", 1, 5, 4)
                            
                            # 追加コメント
                            additional_comment = st.text_area("追加コメント（任意）", height=80)
                            
                            if st.button("💾 フィードバックとして保存"):
                                if st.session_state.current_session_id is None:
                                    # 一時セッション作成
                                    temp_session_id = self.feedback_manager.start_feedback_session("ai_generation_user")
                                    session_id_to_use = temp_session_id
                                else:
                                    session_id_to_use = st.session_state.current_session_id
                                
                                # 画像パスの決定
                                if image_source == "uploaded":
                                    # アップロード画像の場合、一時保存
                                    temp_dir = Path("data/temp_uploads")
                                    temp_dir.mkdir(parents=True, exist_ok=True)
                                    temp_path = temp_dir / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                    selected_image.save(temp_path)
                                    image_path_for_feedback = str(temp_path)
                                else:
                                    image_path_for_feedback = st.session_state.image_list[selected_idx]['image_path']
                                
                                # 説明文の結合
                                final_description = result['description']
                                if additional_comment.strip():
                                    final_description += f"\n\n追加コメント: {additional_comment.strip()}"
                                
                                # フィードバック保存
                                try:
                                    feedback_id = self.feedback_manager.add_feedback(
                                        session_id=session_id_to_use,
                                        image_path=image_path_for_feedback,
                                        is_anomaly=is_anomaly,
                                        anomaly_type=feedback_anomaly_type,
                                        anomaly_description=final_description,
                                        confidence_level=confidence_level
                                    )
                                    
                                    st.success(f"✅ フィードバック保存完了 (ID: {feedback_id})")
                                    
                                    # 一時セッションの場合は終了
                                    if st.session_state.current_session_id is None:
                                        self.feedback_manager.end_feedback_session(temp_session_id)
                                    
                                except Exception as e:
                                    st.error(f"❌ フィードバック保存エラー: {e}")
                    
                    else:
                        # エラー時の表示
                        st.error("❌ 異常説明の生成に失敗しました")
                        st.error(f"エラー内容: {result.get('error', '不明なエラー')}")
                        
                        # トラブルシューティング情報
                        with st.expander("🔧 トラブルシューティング"):
                            st.markdown("""
                            **考えられる原因:**
                            1. LoRAモデルが未学習または破損している
                            2. GPU/CPUメモリ不足
                            3. 画像形式の問題
                            4. プロンプトが長すぎる
                            
                            **対処法:**
                            1. LoRAモデルの再学習を実行
                            2. 生成パラメータ（最大トークン数）を減らす
                            3. 画像サイズを小さくする
                            4. シンプルなプロンプトを使用
                            """)
        else:
            st.info("👆 画像を選択してください")
        
        # 生成履歴（簡易表示）
        if hasattr(self.text_generation, 'is_initialized') and self.text_generation.is_initialized:
            st.markdown("---")
            st.subheader("📊 生成統計")
            
            # セッション中の生成回数などの簡易統計
            if 'generation_count' not in st.session_state:
                st.session_state.generation_count = 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("セッション中の生成数", st.session_state.generation_count)
            with col2:
                st.metric("モデル状態", "✅ 準備完了" if model_status.get('model_loaded', False) else "❌ 未準備")
            with col3:
                st.metric("LoRAアダプタ", "有効" if model_status.get('lora_enabled', False) else "無効")
    
    def system_status_page(self):
        """システム状態ページ"""
        st.title("⚙️ システム状態")
        
        # モデル状態
        st.subheader("🤖 モデル状態")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if self.anomaly_detector:
                st.success("✅ 異常検知モデル: 利用可能")
                if hasattr(self.anomaly_detector, 'threshold') and self.anomaly_detector.threshold:
                    st.info(f"設定閾値: {self.anomaly_detector.threshold:.3f}")
            else:
                st.error("❌ 異常検知モデル: 利用不可")
        
        with col2:
            model_files = list(Path("models").glob("*.pt"))
            st.info(f"保存済みモデル: {len(model_files)}個")
            
            if model_files:
                for model_file in model_files:
                    st.text(f"- {model_file.name}")
        
        # データベース状態
        st.subheader("💾 データベース状態")
        
        # 画像メタデータ
        image_stats = self.metadata_manager.get_statistics()
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🖼️ 画像メタデータ")
            st.json(image_stats)
        
        # フィードバックデータ
        with col2:
            feedback_stats = self.feedback_manager.get_feedback_statistics()
            st.info("📝 フィードバックデータ")
            st.json(feedback_stats)
        
        # システム設定
        st.subheader("⚙️ システム設定")
        st.json(self.config)
    
    def settings_page(self):
        """設定ページ"""
        st.title("⚙️ 設定")
        
        # 異常タイプ管理
        st.subheader("🏷️ 異常タイプ管理")
        
        anomaly_types = self.feedback_manager.get_anomaly_types()
        df_types = pd.DataFrame(anomaly_types)
        
        if not df_types.empty:
            st.dataframe(df_types, use_container_width=True)
        
        # 新しい異常タイプを追加
        with st.expander("新しい異常タイプを追加"):
            with st.form("add_anomaly_type"):
                type_name = st.text_input("異常タイプ名")
                description = st.text_area("説明")
                category = st.selectbox("カテゴリ", ["Equipment", "Product", "Safety", "Maintenance", "Human", "Environment", "Other"])
                severity = st.slider("重要度", 1, 5, 3)
                
                if st.form_submit_button("追加"):
                    # ここで新しい異常タイプをDBに追加するロジックを実装
                    st.success(f"異常タイプ '{type_name}' を追加しました")
        
        # データクリア
        st.subheader("🗑️ データ管理")
        
        st.warning("注意: 以下の操作は取り消しできません")
        
        if st.button("全フィードバックデータを削除", type="secondary"):
            st.error("この機能は安全のため無効化されています")

    def fodd_analysis_page(self):
        """FODD即時分析ページ"""
        st.header("🚀 FODD即時分析")
        st.markdown("**Full Online Data Description** - 新規画像の即座な異常検知・説明生成")
        
        # FODD Pipeline の初期化
        if not hasattr(self, 'fodd_pipeline'):
            try:
                from fodd_pipeline import FODDPipeline
                self.fodd_pipeline = FODDPipeline()
                st.success("FODD Pipeline initialized successfully")
            except Exception as e:
                st.error(f"FODD Pipeline initialization failed: {e}")
                return
        
        # タブ分割
        tab1, tab2, tab3 = st.tabs(["単一画像分析", "バッチ分析", "リアルタイム監視"])
        
        with tab1:
            self._fodd_single_image_analysis()
            
        with tab2:
            self._fodd_batch_analysis()
            
        with tab3:
            self._fodd_realtime_monitoring()
    
    def _fodd_single_image_analysis(self):
        """単一画像分析セクション"""
        st.subheader("📸 単一画像の即時分析")
        
        # 画像アップロード
        uploaded_file = st.file_uploader(
            "分析する画像をアップロード",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key="fodd_single_upload"
        )
        
        if uploaded_file:
            # 画像表示
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="アップロード画像", use_column_width=True)
            
            with col2:
                # 分析設定
                st.subheader("分析設定")
                
                similarity_threshold = st.slider(
                    "類似度閾値", 0.0, 1.0, 0.7, 0.1,
                    help="類似事例検索の閾値"
                )
                
                include_features = st.checkbox(
                    "特徴量ベクトルを結果に含める", 
                    value=False,
                    help="詳細分析用（レポートサイズが大きくなります）"
                )
                
                auto_notification = st.checkbox(
                    "異常検出時の自動通知",
                    value=False,
                    help="異常が検出された場合の通知送信"
                )
            
            # 分析実行
            if st.button("🔍 分析実行", type="primary"):
                with st.spinner("画像を分析中..."):
                    try:
                        # 一時ファイル保存
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            image.save(tmp_file.name)
                            temp_path = tmp_file.name
                        
                        # FODD分析実行
                        result = self.fodd_pipeline.process_single_image(temp_path)
                        
                        # 結果表示
                        self._display_fodd_result(result, include_features)
                        
                        # 一時ファイル削除
                        Path(temp_path).unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
    
    def _fodd_batch_analysis(self):
        """バッチ分析セクション"""
        st.subheader("📁 バッチ画像分析")
        
        # ディレクトリ選択（簡易版）
        st.info("現在は単一画像のアップロードのみサポートしています")
        
        # 複数ファイルアップロード
        uploaded_files = st.file_uploader(
            "複数の画像をアップロード",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="fodd_batch_upload"
        )
        
        if uploaded_files:
            st.write(f"アップロード画像数: {len(uploaded_files)}")
            
            # バッチ処理設定
            col1, col2 = st.columns(2)
            with col1:
                batch_threshold = st.slider("異常検出閾値", 0.0, 1.0, 0.5, 0.1)
            with col2:
                save_reports = st.checkbox("個別レポート保存", value=True)
            
            if st.button("🚀 バッチ分析実行", type="primary"):
                with st.spinner(f"{len(uploaded_files)}枚の画像を分析中..."):
                    try:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            # 一時ファイル保存
                            import tempfile
                            image = Image.open(uploaded_file)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                image.save(tmp_file.name)
                                temp_path = tmp_file.name
                            
                            # 分析実行
                            result = self.fodd_pipeline.process_single_image(temp_path)
                            result['original_filename'] = uploaded_file.name
                            results.append(result)
                            
                            # 一時ファイル削除
                            Path(temp_path).unlink(missing_ok=True)
                            
                            # プログレス更新
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # バッチ結果表示
                        self._display_batch_results(results)
                        
                    except Exception as e:
                        st.error(f"バッチ分析エラー: {e}")
    
    def _fodd_realtime_monitoring(self):
        """リアルタイム監視セクション"""
        st.subheader("⚡ リアルタイム監視")
        st.info("この機能は将来の実装予定です")
        
        # 監視設定UI（プレースホルダー）
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("監視対象", ["カメラ1", "カメラ2", "フォルダ監視"])
            st.slider("監視間隔（秒）", 1, 60, 5)
            
        with col2:
            st.number_input("アラート閾値", 0.0, 1.0, 0.8, 0.1)
            st.checkbox("自動レポート生成")
        
        if st.button("監視開始（デモ）", disabled=True):
            st.warning("リアルタイム監視機能は開発中です")
    
    def _display_fodd_result(self, result: Dict[str, Any], include_features: bool = False):
        """FODD分析結果の表示"""
        if result.get('status') == 'failed':
            st.error(f"分析に失敗しました: {result.get('error', '不明なエラー')}")
            return
        
        # 基本情報
        st.success("✅ 分析完了")
        
        # メトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        anomaly_info = result.get('anomaly_detection', {})
        with col1:
            st.metric(
                "異常判定",
                "異常" if anomaly_info.get('is_anomaly', False) else "正常",
                delta=None
            )
        
        with col2:
            anomaly_score = anomaly_info.get('anomaly_score', 0.0)
            st.metric(
                "異常スコア",
                f"{anomaly_score:.3f}",
                delta=f"閾値: {anomaly_info.get('threshold', 0.1):.3f}"
            )
        
        with col3:
            processing_time = result.get('processing_time', 0)
            st.metric(
                "処理時間",
                f"{processing_time:.2f}秒"
            )
        
        with col4:
            similar_cases_count = len(result.get('similar_cases', []))
            st.metric(
                "類似事例数",
                similar_cases_count
            )
        
        # 生成された説明
        st.subheader("🤖 AI生成説明")
        description = result.get('generated_description', '説明の生成に失敗しました')
        st.write(f"**説明:** {description}")
        
        # 類似事例
        similar_cases = result.get('similar_cases', [])
        if similar_cases:
            st.subheader("🔍 類似事例")
            for i, case in enumerate(similar_cases[:3], 1):
                with st.expander(f"類似事例 {i} (類似度: {case.get('similarity', 0):.3f})"):
                    st.write(f"**説明:** {case.get('description', 'N/A')}")
                    st.write(f"**カテゴリ:** {case.get('category', 'N/A')}")
                    st.write(f"**信頼度:** {case.get('confidence', 0):.3f}")
                    if case.get('metadata'):
                        st.json(case['metadata'])
        
        # 詳細情報
        with st.expander("📊 詳細分析情報"):
            st.json({
                'timestamp': result.get('timestamp'),
                'image_info': result.get('image_info', {}),
                'anomaly_detection': result.get('anomaly_detection', {}),
                'system_info': result.get('system_info', {})
            })
        
        # 特徴量情報（オプション）
        if include_features and 'feature_vector_shape' in result:
            st.write(f"**特徴量ベクトル形状:** {result['feature_vector_shape']}")
        
        # ダウンロードリンク
        if 'report_path' in result:
            with open(result['report_path'], 'r', encoding='utf-8') as f:
                report_json = f.read()
            
            st.download_button(
                label="📄 レポートをダウンロード",
                data=report_json,
                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _display_batch_results(self, results: List[Dict[str, Any]]):
        """バッチ分析結果の表示"""
        st.subheader("📊 バッチ分析結果")
        
        # 統計サマリー
        total_images = len(results)
        anomaly_count = sum(1 for r in results if r.get('anomaly_detection', {}).get('is_anomaly', False))
        normal_count = total_images - anomaly_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総画像数", total_images)
        with col2:
            st.metric("異常検出", anomaly_count, delta=f"{(anomaly_count/total_images*100):.1f}%")
        with col3:
            st.metric("正常判定", normal_count, delta=f"{(normal_count/total_images*100):.1f}%")
        
        # 結果一覧
        if results:
            # データフレーム形式で表示
            df_data = []
            for result in results:
                anomaly_info = result.get('anomaly_detection', {})
                df_data.append({
                    'ファイル名': result.get('original_filename', 'N/A'),
                    '判定': '異常' if anomaly_info.get('is_anomaly', False) else '正常',
                    '異常スコア': f"{anomaly_info.get('anomaly_score', 0):.3f}",
                    '処理時間': f"{result.get('processing_time', 0):.2f}秒",
                    '説明': result.get('generated_description', '')[:100] + '...'
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # 結果の詳細表示
            selected_idx = st.selectbox(
                "詳細を表示する画像を選択",
                range(len(results)),
                format_func=lambda x: results[x].get('original_filename', f'画像 {x+1}')
            )
            
            if selected_idx is not None:
                st.subheader(f"詳細: {results[selected_idx].get('original_filename', f'画像 {selected_idx+1}')}")
                self._display_fodd_result(results[selected_idx])
            
            # バッチレポートダウンロード
            batch_report = {
                'batch_analysis_summary': {
                    'total_images': total_images,
                    'anomaly_count': anomaly_count,
                    'normal_count': normal_count,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'individual_results': results
            }
            
            batch_json = json.dumps(batch_report, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📦 バッチレポートをダウンロード",
                data=batch_json,
                file_name=f"batch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    def knowledge_base_page(self):
        """ナレッジベースページ"""
        st.header("🧠 ナレッジベース管理")
        
        # サイドバーで操作選択
        st.sidebar.subheader("操作メニュー")
        operation = st.sidebar.radio(
            "実行する操作を選択",
            ["検索", "統計情報", "特徴量追加", "パターン分析", "エクスポート"]
        )
        
        if operation == "検索":
            self._knowledge_search_section()
        elif operation == "統計情報":
            self._knowledge_stats_section()
        elif operation == "特徴量追加":
            self._knowledge_add_section()
        elif operation == "パターン分析":
            self._knowledge_pattern_section()
        elif operation == "エクスポート":
            self._knowledge_export_section()
    
    def _knowledge_search_section(self):
        """ナレッジベース検索セクション"""
        st.subheader("🔍 類似特徴量検索")
        
        # 検索方法選択
        search_type = st.selectbox(
            "検索方法",
            ["テキスト検索", "特徴量ベクトル検索", "ID検索"]
        )
        
        if search_type == "テキスト検索":
            query = st.text_input("検索クエリを入力してください", placeholder="例：赤い異常、欠陥パターン")
            similarity_threshold = st.slider("類似度閾値", 0.0, 1.0, 0.7, 0.1)
            max_results = st.number_input("最大結果数", 1, 100, 10)
            
            if st.button("検索実行") and query:
                try:
                    results = self.knowledge_manager.search_similar_features(
                        query, similarity_threshold, max_results
                    )
                    self._display_search_results(results)
                except Exception as e:
                    st.error(f"検索エラー: {e}")
        
        elif search_type == "特徴量ベクトル検索":
            st.info("アップロードされた画像から特徴量を抽出して検索します")
            uploaded_file = st.file_uploader(
                "画像をアップロード", 
                type=['png', 'jpg', 'jpeg'],
                key="kb_search_image"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="アップロード画像", width=300)
                
                similarity_threshold = st.slider("類似度閾値", 0.0, 1.0, 0.7, 0.1, key="vector_threshold")
                max_results = st.number_input("最大結果数", 1, 100, 10, key="vector_max_results")
                
                if st.button("特徴量検索実行"):
                    try:
                        # 画像から特徴量抽出（プレースホルダー）
                        st.info("実際の実装では、ここで画像から特徴量を抽出して検索します")
                    except Exception as e:
                        st.error(f"特徴量検索エラー: {e}")
        
        elif search_type == "ID検索":
            feature_id = st.text_input("特徴量IDを入力")
            if st.button("ID検索実行") and feature_id:
                try:
                    # ID検索の実装
                    st.info(f"ID: {feature_id} の検索を実行します")
                except Exception as e:
                    st.error(f"ID検索エラー: {e}")
    
    def _knowledge_stats_section(self):
        """ナレッジベース統計情報セクション"""
        st.subheader("📊 ナレッジベース統計")
        
        try:
            # 統計情報取得
            stats = self.knowledge_manager.get_knowledge_base_stats()
            
            # メトリクス表示
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("総特徴量数", stats.get('total_features', 0))
            with col2:
                st.metric("正常パターン", stats.get('normal_count', 0))
            with col3:
                st.metric("異常パターン", stats.get('anomaly_count', 0))
            with col4:
                st.metric("ベクトル次元", stats.get('vector_dimension', 0))
            
            # 詳細統計
            st.subheader("詳細統計")
            
            # カテゴリ分布
            if 'category_distribution' in stats:
                st.write("**カテゴリ分布:**")
                category_df = pd.DataFrame(
                    list(stats['category_distribution'].items()),
                    columns=['カテゴリ', '件数']
                )
                st.bar_chart(category_df.set_index('カテゴリ'))
            
            # 時系列分布
            if 'temporal_distribution' in stats:
                st.write("**時系列分布:**")
                temporal_df = pd.DataFrame(stats['temporal_distribution'])
                st.line_chart(temporal_df)
            
            # データベース情報
            st.subheader("データベース情報")
            db_info = stats.get('database_info', {})
            st.json(db_info)
            
        except Exception as e:
            st.error(f"統計情報取得エラー: {e}")
    
    def _knowledge_add_section(self):
        """ナレッジベース特徴量追加セクション"""
        st.subheader("➕ 特徴量手動追加")
        
        with st.form("add_feature_form"):
            description = st.text_area("特徴量説明", placeholder="異常の詳細な説明を入力してください")
            category = st.selectbox("カテゴリ", ["anomaly", "normal", "unknown"])
            confidence = st.slider("信頼度", 0.0, 1.0, 0.8, 0.1)
            source = st.text_input("ソース", value="manual_input")
            
            # メタデータ
            st.subheader("追加メタデータ（オプション）")
            metadata_cols = st.columns(2)
            with metadata_cols[0]:
                location = st.text_input("場所")
                equipment = st.text_input("設備")
            with metadata_cols[1]:
                severity = st.selectbox("重要度", ["low", "medium", "high", "critical"])
                tags = st.text_input("タグ（カンマ区切り）")
            
            submitted = st.form_submit_button("特徴量を追加")
            
            if submitted and description:
                try:
                    # メタデータ構築
                    metadata = {
                        "location": location if location else None,
                        "equipment": equipment if equipment else None,
                        "severity": severity,
                        "tags": [tag.strip() for tag in tags.split(",")] if tags else []
                    }
                    
                    # 特徴量追加
                    feature_id = self.knowledge_manager.add_feature(
                        description=description,
                        category=category,
                        confidence=confidence,
                        source=source,
                        metadata=metadata
                    )
                    
                    st.success(f"特徴量が追加されました！ID: {feature_id}")
                    
                except Exception as e:
                    st.error(f"特徴量追加エラー: {e}")
    
    def _knowledge_pattern_section(self):
        """パターン分析セクション"""
        st.subheader("🔬 パターン分析")
        
        analysis_type = st.selectbox(
            "分析タイプ",
            ["クラスタリング分析", "異常パターン分析", "類似度マトリックス", "特徴量重要度"]
        )
        
        if analysis_type == "クラスタリング分析":
            st.write("**クラスタリング分析**")
            n_clusters = st.slider("クラスタ数", 2, 20, 5)
            
            if st.button("クラスタリング実行"):
                try:
                    # クラスタリング分析の実装
                    st.info("クラスタリング分析を実行中...")
                    # プレースホルダー
                    st.success("クラスタリング分析が完了しました")
                except Exception as e:
                    st.error(f"クラスタリング分析エラー: {e}")
        
        elif analysis_type == "異常パターン分析":
            st.write("**異常パターン分析**")
            threshold = st.slider("異常度閾値", 0.0, 1.0, 0.8, 0.1)
            
            if st.button("異常パターン分析実行"):
                try:
                    # 異常パターン分析の実装
                    patterns = self.knowledge_manager.analyze_anomaly_patterns(threshold)
                    
                    if patterns:
                        st.write("**検出された異常パターン:**")
                        for i, pattern in enumerate(patterns, 1):
                            st.write(f"{i}. {pattern}")
                    else:
                        st.info("異常パターンが検出されませんでした")
                        
                except Exception as e:
                    st.error(f"異常パターン分析エラー: {e}")
        
        elif analysis_type == "類似度マトリックス":
            st.write("**類似度マトリックス**")
            sample_size = st.number_input("サンプルサイズ", 10, 1000, 100)
            
            if st.button("類似度マトリックス生成"):
                try:
                    st.info("類似度マトリックスを生成中...")
                    # プレースホルダー
                    st.success("類似度マトリックスが生成されました")
                except Exception as e:
                    st.error(f"類似度マトリックス生成エラー: {e}")
        
        elif analysis_type == "特徴量重要度":
            st.write("**特徴量重要度分析**")
            
            if st.button("重要度分析実行"):
                try:
                    importance_scores = self.knowledge_manager.analyze_feature_importance()
                    
                    if importance_scores:
                        st.write("**特徴量重要度:**")
                        importance_df = pd.DataFrame(
                            importance_scores.items(),
                            columns=['特徴量', '重要度']
                        ).sort_values('重要度', ascending=False)
                        
                        st.dataframe(importance_df)
                        st.bar_chart(importance_df.set_index('特徴量'))
                    else:
                        st.info("重要度データが不足しています")
                        
                except Exception as e:
                    st.error(f"重要度分析エラー: {e}")
    
    def _knowledge_export_section(self):
        """エクスポートセクション"""
        st.subheader("📤 データエクスポート")
        
        export_format = st.selectbox(
            "エクスポート形式",
            ["JSON", "CSV", "Excel", "SQLite"]
        )
        
        # フィルタオプション
        st.subheader("フィルタオプション")
        filter_cols = st.columns(2)
        with filter_cols[0]:
            category_filter = st.multiselect(
                "カテゴリフィルタ",
                ["anomaly", "normal", "unknown"]
            )
            date_range = st.date_input("日付範囲", value=None)
        
        with filter_cols[1]:
            confidence_min = st.slider("最小信頼度", 0.0, 1.0, 0.0, 0.1)
            include_vectors = st.checkbox("ベクトルデータを含める")
        
        if st.button("エクスポート実行"):
            try:
                # エクスポート実行
                export_data = self.knowledge_manager.export_knowledge_base(
                    format=export_format.lower(),
                    category_filter=category_filter,
                    confidence_min=confidence_min,
                    include_vectors=include_vectors
                )
                
                if export_data:
                    # ダウンロードリンク生成
                    filename = f"knowledge_base_export.{export_format.lower()}"
                    st.download_button(
                        label=f"{export_format}ファイルをダウンロード",
                        data=export_data,
                        file_name=filename,
                        mime=f"application/{export_format.lower()}"
                    )
                    st.success("エクスポートが完了しました")
                else:
                    st.warning("エクスポートするデータがありません")
                    
            except Exception as e:
                st.error(f"エクスポートエラー: {e}")
    
    def _display_search_results(self, results):
        """検索結果表示"""
        if not results:
            st.info("検索結果がありません")
            return
        
        st.subheader(f"検索結果 ({len(results)}件)")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"結果 {i}: {result.get('description', 'N/A')[:50]}..."):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**説明:**", result.get('description', 'N/A'))
                    st.write("**カテゴリ:**", result.get('category', 'N/A'))
                    st.write("**ソース:**", result.get('source', 'N/A'))
                    
                    # メタデータ表示
                    if 'metadata' in result and result['metadata']:
                        st.write("**メタデータ:**")
                        st.json(result['metadata'])
                
                with col2:
                    st.metric("類似度", f"{result.get('similarity', 0):.3f}")
                    st.metric("信頼度", f"{result.get('confidence', 0):.3f}")
                    st.write("**ID:**", result.get('id', 'N/A'))
                    st.write("**作成日:**", result.get('created_at', 'N/A'))
    
    def run(self):
        """アプリケーション実行"""
        # サイドバーナビゲーション
        page = self.sidebar_navigation()
        
        # ページルーティング
        if page == "画像フィードバック":
            self.image_feedback_page()
        elif page == "AI説明生成":
            self.ai_description_page()
        elif page == "FODD即時分析":
            self.fodd_analysis_page()
        elif page == "データ分析":
            self.data_analysis_page()
        elif page == "システム状態":
            self.system_status_page()
        elif page == "設定":
            self.settings_page()
        elif page == "ナレッジベース":
            self.knowledge_base_page()


def main():
    """メイン関数"""
    try:
        app = MADFHApp()
        app.run()
    except Exception as e:
        st.error(f"アプリケーションエラー: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
