"""
Windows PowerShellの文字エンコーディング対応のため、
Unicode絵文字を削除して英数字のみにする
"""

import re

def fix_unicode_in_file(filepath):
    """ファイル内のUnicode絵文字を修正"""
    
    # 絵文字と記号の置換マップ
    replacements = {
        '🚀': '',
        '✅': '[OK]',
        '❌': '[FAILED]',
        '📊': '[REPORT]',
        '🔧': '[GPU]',
        '⚠️': '[WARNING]',
        '💡': '[TIP]',
        '🔍': '[CHECK]',
        '📋': '[CONFIG]',
        '🎯': ''
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 絵文字置換
        for emoji, replacement in replacements.items():
            content = content.replace(emoji, replacement)
        
        # 余分なスペースを整理
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r' +\n', '\n', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed: {filepath}")
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

if __name__ == "__main__":
    # 対象ファイルリスト
    files_to_fix = [
        "train_wallplugs_integrated.py"
    ]
    
    for file in files_to_fix:
        fix_unicode_in_file(file)
    
    print("Unicode fix completed!")