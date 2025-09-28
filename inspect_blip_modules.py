"""
BLIPモデルの構造とLoRA対象モジュールを確認するスクリプト
"""

from transformers import BlipForConditionalGeneration
import torch

def inspect_blip_model():
    """BLIPモデル構造を調査"""
    print("BLIPモデル構造調査中...")
    
    try:
        # BLIPモデル読み込み
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        
        print("モデル読み込み成功!")
        print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 全モジュール名を収集
        module_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                module_names.append(name)
        
        # 線形層（Linear）を特定
        linear_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append(name)
        
        print(f"\n線形層（LoRA対象候補）:")
        for i, name in enumerate(linear_modules[:20]):  # 最初の20個表示
            print(f"  {i+1:2d}. {name}")
        
        if len(linear_modules) > 20:
            print(f"     ... 他 {len(linear_modules)-20} 個")
        
        # 推奨 target_modules
        common_patterns = []
        for name in linear_modules:
            # q_proj, k_proj, v_proj, out_proj パターン検索
            if any(pattern in name for pattern in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                common_patterns.append(name)
            # query, key, value パターン検索
            elif any(pattern in name for pattern in ['query', 'key', 'value']):
                common_patterns.append(name)
            # fc, dense パターン検索
            elif any(pattern in name for pattern in ['fc', 'dense']):
                common_patterns.append(name)
        
        print(f"\nLoRA推奨 target_modules:")
        if common_patterns:
            unique_types = set()
            for name in common_patterns[:10]:  # 最初の10個
                module_type = name.split('.')[-1]  # 最後の部分取得
                unique_types.add(module_type)
                print(f"  - {name}")
            
            print(f"\n推奨設定 (モジュールタイプ):")
            print(f"target_modules = {list(unique_types)}")
        else:
            # フォールバック: 一般的な線形層
            fallback_modules = [name.split('.')[-1] for name in linear_modules[:5]]
            unique_fallback = list(set(fallback_modules))
            print(f"target_modules = {unique_fallback}")
        
        return list(unique_types) if common_patterns else unique_fallback
        
    except Exception as e:
        print(f"エラー: {e}")
        return []

if __name__ == "__main__":
    target_modules = inspect_blip_model()
    print(f"\n🎯 結論: target_modules = {target_modules}")