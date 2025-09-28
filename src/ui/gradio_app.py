"""
MAD-FH: Gradio Human-in-the-Loop UI
簡易版の人間フィードバック収集インターフェース
"""

import gradio as gr
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import json
import torch
from datetime import datetime
import yaml

# プロジェクト内のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ui.feedback_manager import FeedbackDataManager
from src.data.metadata_manager import ImageMetadataDB
from src.models.autoencoder import ConvAutoencoder, AnomalyDetector
from src.data.preprocess import ImagePreprocessor


class MADFHGradioApp:
    """MAD-FH Gradioアプリケーション"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.load_config()
        self.initialize_managers()
        self.current_session_id = None
        self.current_image_index = 0
        self.image_list = []
        self.load_image_list()
    
    def load_config(self):
        """設定ファイルの読み込み"""
        config_path = "config/config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
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
        
        # 異常検知器
        self.anomaly_detector = None
        self.load_anomaly_detector()
    
    def load_anomaly_detector(self):
        """異常検知器の読み込み"""
        model_path = "models/best_autoencoder_model.pt"
        
        if Path(model_path).exists():
            try:
                model = ConvAutoencoder(
                    input_channels=3,
                    latent_dim=256,
                    input_size=(512, 512)
                )
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.anomaly_detector = AnomalyDetector(model, device='cpu')
                print("異常検知モデルを読み込みました")
            except Exception as e:
                print(f"モデル読み込みエラー: {e}")
    
    def load_image_list(self):
        """画像リストの読み込み"""
        images = self.metadata_manager.list_images()
        self.image_list = [img for img in images if Path(img['image_path']).exists()]
        print(f"読み込み画像数: {len(self.image_list)}")
    
    def start_session(self, user_id):
        """セッション開始"""
        if not user_id.strip():
            user_id = "default_user"
        
        self.current_session_id = self.feedback_manager.start_feedback_session(user_id)
        return f"セッション開始: {self.current_session_id[:8]}...", gr.update(interactive=False), gr.update(interactive=True)
    
    def end_session(self):
        """セッション終了"""
        if self.current_session_id:
            self.feedback_manager.end_feedback_session(self.current_session_id)
            session_id = self.current_session_id
            self.current_session_id = None
            return f"セッション終了: {session_id[:8]}...", gr.update(interactive=True), gr.update(interactive=False)
        return "アクティブなセッションがありません", gr.update(), gr.update()
    
    def get_current_image_info(self):
        """現在の画像情報を取得"""
        if not self.image_list or self.current_image_index >= len(self.image_list):
            return None, "画像が見つかりません", "", 0.0, ""
        
        current_image = self.image_list[self.current_image_index]
        image_path = current_image['image_path']
        
        try:
            # 画像読み込み
            image = Image.open(image_path)
            
            # 画像情報
            info = f"""
            ファイル名: {current_image['filename']}
            撮影日時: {current_image.get('capture_datetime', 'N/A')}
            カメラID: {current_image.get('camera_id', 'N/A')}
            場所: {current_image.get('location', 'N/A')}
            サイズ: {image.size}
            """
            
            # AI異常検知
            ai_result = ""
            ai_score = 0.0
            
            if self.anomaly_detector:
                try:
                    tensor = self.preprocessor.preprocess_image(image_path, apply_augmentation=False)
                    tensor = tensor.unsqueeze(0)
                    
                    scores, predictions = self.anomaly_detector.predict(tensor)
                    score = scores[0]
                    prediction = predictions[0]
                    
                    ai_score = float(score)
                    if prediction == 1:
                        ai_result = f"🚨 AI判定: 異常 (スコア: {score:.3f})"
                    else:
                        ai_result = f"✅ AI判定: 正常 (スコア: {score:.3f})"
                        
                except Exception as e:
                    ai_result = f"AI判定エラー: {e}"
            else:
                ai_result = "AI異常検知モデルが利用できません"
            
            navigation = f"画像 {self.current_image_index + 1} / {len(self.image_list)}"
            
            return image, info.strip(), ai_result, ai_score, navigation
            
        except Exception as e:
            return None, f"画像読み込みエラー: {e}", "", 0.0, ""
    
    def navigate_image(self, direction):
        """画像ナビゲーション"""
        if direction == "previous" and self.current_image_index > 0:
            self.current_image_index -= 1
        elif direction == "next" and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
        
        return self.get_current_image_info()
    
    def submit_feedback(self, is_anomaly, anomaly_type, description, confidence):
        """フィードバック送信"""
        if not self.current_session_id:
            return "エラー: セッションが開始されていません"
        
        if not self.image_list or self.current_image_index >= len(self.image_list):
            return "エラー: 有効な画像がありません"
        
        current_image = self.image_list[self.current_image_index]
        
        try:
            # フィードバック保存
            feedback_id = self.feedback_manager.add_feedback(
                image_path=current_image['image_path'],
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type if is_anomaly else None,
                anomaly_description=description if is_anomaly else "",
                confidence_level=confidence,
                session_id=self.current_session_id,
                image_id=current_image.get('id')
            )
            
            # 次の画像に自動移動
            if self.current_image_index < len(self.image_list) - 1:
                self.current_image_index += 1
            
            result = f"✅ フィードバックを保存しました！ (ID: {feedback_id[:8]}...)"
            
            # 次の画像情報を取得
            image, info, ai_result, ai_score, navigation = self.get_current_image_info()
            
            return result, image, info, ai_result, navigation
            
        except Exception as e:
            return f"エラー: {e}", gr.update(), gr.update(), gr.update(), gr.update()
    
    def get_statistics(self):
        """統計情報取得"""
        stats = self.feedback_manager.get_feedback_statistics()
        
        total = stats.get('total_feedbacks', 0)
        sessions = stats.get('total_sessions', 0)
        
        anomaly_ratio = stats.get('anomaly_normal_ratio', {})
        anomaly_count = anomaly_ratio.get(1, 0)
        normal_count = anomaly_ratio.get(0, 0)
        
        anomaly_types = stats.get('anomaly_type_statistics', {})
        top_anomaly_types = sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = f"""
        ## 📊 統計情報
        
        - **総フィードバック数**: {total}
        - **セッション数**: {sessions}
        - **正常**: {normal_count} / **異常**: {anomaly_count}
        - **異常率**: {(anomaly_count / total * 100) if total > 0 else 0:.1f}%
        
        ### トップ異常タイプ:
        """
        
        for i, (anomaly_type, count) in enumerate(top_anomaly_types, 1):
            result += f"{i}. {anomaly_type}: {count}件\n"
        
        return result.strip()
    
    def create_interface(self):
        """Gradioインターフェースの作成"""
        
        # 異常タイプ選択肢
        anomaly_types = [at['type_name'] for at in self.feedback_manager.get_anomaly_types()]
        
        with gr.Blocks(title="MAD-FH: Multimodal Anomaly Detector", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 🔍 MAD-FH: Multimodal Anomaly Detector with Human Feedback")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 画像表示エリア
                    image_display = gr.Image(label="検査対象画像", height=400)
                    
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ 前の画像", variant="secondary")
                        navigation_info = gr.Textbox(label="ナビゲーション", interactive=False)
                        next_btn = gr.Button("次の画像 ➡️", variant="secondary")
                    
                    # 画像情報
                    image_info = gr.Textbox(label="画像情報", lines=5, interactive=False)
                    
                    # AI判定結果
                    ai_result = gr.Textbox(label="AI異常検知結果", interactive=False)
                
                with gr.Column(scale=1):
                    # セッション管理
                    gr.Markdown("## 📝 セッション管理")
                    
                    user_id_input = gr.Textbox(label="ユーザーID", value="operator_01")
                    session_status = gr.Textbox(label="セッション状態", interactive=False)
                    
                    with gr.Row():
                        start_session_btn = gr.Button("セッション開始", variant="primary")
                        end_session_btn = gr.Button("セッション終了", variant="secondary", interactive=False)
                    
                    # フィードバック入力
                    gr.Markdown("## 👤 人間フィードバック")
                    
                    is_anomaly = gr.Radio(
                        choices=[("正常", False), ("異常", True)],
                        label="判定結果",
                        value=False
                    )
                    
                    anomaly_type = gr.Dropdown(
                        choices=anomaly_types,
                        label="異常タイプ",
                        interactive=False
                    )
                    
                    description = gr.Textbox(
                        label="異常詳細説明",
                        placeholder="具体的な異常内容を記述...",
                        lines=3,
                        interactive=False
                    )
                    
                    confidence = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="確信度 (1:不確実 ～ 5:確実)"
                    )
                    
                    submit_btn = gr.Button("📝 フィードバック送信", variant="primary")
                    feedback_status = gr.Textbox(label="送信結果", interactive=False)
            
            # 統計情報タブ
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📊 統計情報")
                    stats_btn = gr.Button("統計更新", variant="secondary")
                    stats_display = gr.Markdown()
            
            # イベントハンドラー
            
            # セッション管理
            start_session_btn.click(
                fn=self.start_session,
                inputs=[user_id_input],
                outputs=[session_status, start_session_btn, end_session_btn]
            )
            
            end_session_btn.click(
                fn=self.end_session,
                inputs=[],
                outputs=[session_status, start_session_btn, end_session_btn]
            )
            
            # 画像ナビゲーション
            prev_btn.click(
                fn=lambda: self.navigate_image("previous"),
                inputs=[],
                outputs=[image_display, image_info, ai_result, gr.Number(visible=False), navigation_info]
            )
            
            next_btn.click(
                fn=lambda: self.navigate_image("next"),
                inputs=[],
                outputs=[image_display, image_info, ai_result, gr.Number(visible=False), navigation_info]
            )
            
            # 異常判定に応じてフォームを更新
            def update_form(is_anomaly_val):
                return (
                    gr.update(interactive=is_anomaly_val),  # anomaly_type
                    gr.update(interactive=is_anomaly_val)   # description
                )
            
            is_anomaly.change(
                fn=update_form,
                inputs=[is_anomaly],
                outputs=[anomaly_type, description]
            )
            
            # フィードバック送信
            submit_btn.click(
                fn=self.submit_feedback,
                inputs=[is_anomaly, anomaly_type, description, confidence],
                outputs=[feedback_status, image_display, image_info, ai_result, navigation_info]
            )
            
            # 統計情報更新
            stats_btn.click(
                fn=self.get_statistics,
                inputs=[],
                outputs=[stats_display]
            )
            
            # 初期画像読み込み
            interface.load(
                fn=self.get_current_image_info,
                inputs=[],
                outputs=[image_display, image_info, ai_result, gr.Number(visible=False), navigation_info]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """アプリケーション起動"""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """メイン関数"""
    app = MADFHGradioApp()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()
