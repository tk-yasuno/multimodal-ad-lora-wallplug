#!/usr/bin/env python3
"""
BLIPモデルの構造確認スクリプト
"""

from transformers import BlipForConditionalGeneration

def main():
    print("BLIPモデルの構造を確認中...")
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    
    print("\n=== BLIPモデルの主要モジュール ===")
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if any(keyword in name.lower() for keyword in ['attention', 'linear', 'dense', 'projection']):
            print(f"- {name}: {module_type}")

if __name__ == "__main__":
    main()
