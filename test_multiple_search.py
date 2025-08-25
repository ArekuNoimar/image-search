#!/usr/bin/env python3

"""複数検索最適化の自動テスト用スクリプト。

新機能のsearch_similar_image_only関数を使用して、
複数回連続検索のパフォーマンスを自動的にテストします。
"""

import sys
import os
import time
sys.path.append('src')

from main import load_config, search_similar_image_only
from model import ImageEmbeddingModel

def test_multiple_search_optimization():
    """複数検索最適化の自動テストを実行します。
    
    モデルを1回だけ読み込み、複数の画像で連続検索を行って
    パフォーマンスを測定します。
    
    Returns:
        bool: テストが成功した場合True
    """
    print("=== 複数検索最適化自動テスト ===\n")
    
    try:
        # 設定を読み込み
        config = load_config('src/config.json')
        
        # テスト用画像パスリスト
        test_images = [
            'apple.jpg',
            'src/images/banana.jpg',
            'src/images/grapes.jpg'
        ]
        
        # 存在する画像のみフィルター
        available_images = [img for img in test_images if os.path.exists(img)]
        
        if len(available_images) < 2:
            print("❌ テストには最低2個の画像が必要です")
            return False
        
        print(f"テスト対象画像: {len(available_images)}個")
        for img in available_images:
            print(f"  - {img}")
        
        # モデルを1回だけ読み込み
        print("\n1. モデルを読み込み中...")
        start_time = time.time()
        model = ImageEmbeddingModel(config['model-name'], config['device'])
        model_load_time = time.time() - start_time
        print(f"✅ モデル読み込み完了: {model_load_time:.2f}秒")
        
        # 各画像で検索実行
        search_times = []
        
        for i, image_path in enumerate(available_images, 1):
            print(f"\n{i}. 検索実行: {os.path.basename(image_path)}")
            
            start_time = time.time()
            search_similar_image_only(image_path, config, model)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            print(f"   検索時間: {search_time:.2f}秒")
        
        # パフォーマンス結果表示
        print(f"\n=== パフォーマンス結果 ===")
        print(f"モデル読み込み時間: {model_load_time:.2f}秒")
        print(f"総検索回数: {len(search_times)}回")
        print(f"平均検索時間: {sum(search_times)/len(search_times):.2f}秒")
        print(f"最速検索時間: {min(search_times):.2f}秒")
        print(f"最遅検索時間: {max(search_times):.2f}秒")
        
        # 最適化効果の確認
        if len(search_times) > 1:
            first_search = search_times[0]
            subsequent_searches = search_times[1:]
            avg_subsequent = sum(subsequent_searches) / len(subsequent_searches)
            
            print(f"\n=== 最適化効果 ===")
            print(f"初回検索時間: {first_search:.2f}秒")
            print(f"2回目以降平均: {avg_subsequent:.2f}秒")
            
            if avg_subsequent < first_search:
                improvement = ((first_search - avg_subsequent) / first_search) * 100
                print(f"✅ 高速化達成: {improvement:.1f}%向上")
            else:
                print("⚠️  高速化は確認できませんでした")
        
        print(f"\n🎉 複数検索最適化テスト完了!")
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multiple_search_optimization()