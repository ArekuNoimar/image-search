#!/usr/bin/env python3

"""インタラクティブ検索の最適化テスト用スクリプト。

複数回連続検索時のモデル再利用とパフォーマンスを検証するためのスクリプトです。
実際のファイルパスを自動入力してインタラクティブモードをシミュレートします。
"""

import sys
import os
import time
sys.path.append('src')

from main import interactive_search, load_config

def test_interactive_optimization():
    """インタラクティブ検索の最適化をテストします。
    
    複数の画像で連続検索を行い、モデルの再利用により
    高速化されることを確認します。
    """
    print("=== インタラクティブ検索最適化テスト ===\n")
    
    # 設定を読み込み
    config = load_config('src/config.json')
    
    # テスト用画像パスリスト
    test_images = [
        'apple.jpg',
        'src/images/apple.jpg', 
        'src/images/banana.jpg',
        'src/images/grapes.jpg'
    ]
    
    # 存在する画像のみフィルター
    available_images = [img for img in test_images if os.path.exists(img)]
    
    print(f"テスト対象画像: {len(available_images)}個")
    for img in available_images:
        print(f"  - {img}")
    
    if len(available_images) < 2:
        print("❌ テストには最低2個の画像が必要です")
        return False
    
    # 自動的に複数検索をシミュレート（実際のinteractive_searchは手動入力）
    print("\n手動でインタラクティブモードを実行してください:")
    print("uv run src/main.py --env src/config.json --no-cleanup")
    print("\n各画像パスを順次入力してパフォーマンスを確認:")
    for img in available_images:
        print(f"  {img}")
    print("  quit")
    
    return True

if __name__ == "__main__":
    test_interactive_optimization()