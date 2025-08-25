"""画像検索システムのメイン実行モジュール。

このモジュールは画像検索システム全体の実行フローを管理します。
画像の前処理、データベースへの埋め込みベクトル保存、
類似画像検索、結果出力の一連の処理を行います。
"""

import os
import sys
import time
import json
import argparse
import shutil
import signal
import atexit
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from db import DatabaseManager, generate_file_hash, cosine_similarity
from model import ImageEmbeddingModel, process_images_in_directory


def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込みます。
    
    Args:
        config_path (str): 設定ファイルのパス
        
    Returns:
        Dict[str, Any]: 設定内容の辞書
        
    Raises:
        Exception: 設定ファイルの読み込みに失敗した場合
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"設定ファイルを読み込みました: {config_path}")
        return config
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        raise


def create_output_directory(base_path: str) -> str:
    """タイムスタンプ付きの出力ディレクトリを作成します。
    
    Args:
        base_path (str): ベースとなるディレクトリパス
        
    Returns:
        str: 作成された出力ディレクトリのパス
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_search_results(output_dir: str, config: Dict[str, Any], results: List[Dict[str, Any]], 
                       query_image_path: str, best_match_path: str):
    """検索結果を指定されたディレクトリに保存します。
    
    設定ファイル、検索結果リスト、最も類似した画像を
    タイムスタンプ付きディレクトリに保存します。
    
    Args:
        output_dir (str): 出力先ディレクトリのパス
        config (Dict[str, Any]): 使用した設定内容
        results (List[Dict[str, Any]]): 検索結果のリスト
        query_image_path (str): 検索対象画像のパス
        best_match_path (str): 最も類似した画像のパス
    """
    config_filename = f"{os.path.splitext(os.path.basename(query_image_path))[0]}.json"
    result_filename = f"{os.path.splitext(os.path.basename(query_image_path))[0]}.jsonl"
    
    # 設定ファイルを保存
    config_output_path = os.path.join(output_dir, config_filename)
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 検索結果を保存
    result_output_path = os.path.join(output_dir, result_filename)
    with open(result_output_path, 'w') as f:
        for i, result in enumerate(results):
            result_data = {
                "rank": i + 1,
                "similarity": float(result.get('cosine_similarity', 0)),
                "file_name": result['file_name'],
                "file_path": result['file_path']
            }
            f.write(json.dumps(result_data) + '\n')
    
    # 最も類似した画像をコピー
    if best_match_path and os.path.exists(best_match_path):
        best_match_filename = f"{os.path.splitext(os.path.basename(query_image_path))[0]}.jpg"
        best_match_output = os.path.join(output_dir, best_match_filename)
        shutil.copy2(best_match_path, best_match_output)
        print(f"最も類似した画像を保存しました: {best_match_output}")
    
    print(f"結果を保存しました: {output_dir}")


def process_reference_images(config: Dict[str, Any]) -> None:
    """参照画像を処理してデータベースに埋め込みベクトルを保存します。
    
    指定されたディレクトリ内の全ての画像に対して前処理（リサイズ・フォーマット変換）
    を行い、CLIPモデルで埋め込みベクトルを生成してデータベースに保存します。
    重複チェックも行います。
    
    Args:
        config (Dict[str, Any]): システム設定の辞書
    """
    print("参照画像の処理を開始します...")
    
    db_manager = DatabaseManager(
        host=config['postgres-host'],
        port=config['postgres-port'],
        user=config['postgres-user'],
        password=config['postgres-password'],
        database=config['postgres-database']
    )
    
    try:
        db_manager.connect()
        db_manager.create_table()
        
        processed_images = process_images_in_directory(
            config['source-directory'],
            config['processed-directory'],
            config['resize-width'],
            config['resize-height']
        )
        
        if not processed_images:
            print("処理対象の画像がありません")
            return
            
        model = ImageEmbeddingModel(config['model-name'], config['device'])
        
        print("埋め込みベクトルを生成してデータベースに保存中...")
        for image_path in tqdm(processed_images, desc="埋め込みベクトル処理中"):
            try:
                file_hash = generate_file_hash(image_path)
                
                if not db_manager.hash_exists(file_hash):
                    embedding = model.encode_image(image_path)
                    file_name = os.path.basename(image_path)
                    
                    db_manager.insert_embedding(
                        file_path=image_path,
                        file_name=file_name,
                        file_hash=file_hash,
                        embedding=embedding
                    )
                else:
                    print(f"重複ファイルをスキップしました: {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"画像処理エラー {image_path}: {e}")
                
    except Exception as e:
        print(f"データベースエラー: {e}")
        raise
    finally:
        db_manager.close()


def search_similar_image_only(query_image_path: str, config: Dict[str, Any], model: Optional[ImageEmbeddingModel] = None) -> None:
    """クエリ画像のみを処理して類似画像を検索します（参照画像の前処理なし）。
    
    既にデータベースに埋め込みベクトルが保存されている前提で、
    クエリ画像のみを処理して類似度検索を実行します。
    
    Args:
        query_image_path (str): 検索対象画像のパス
        config (Dict[str, Any]): システム設定の辞書
        model (Optional[ImageEmbeddingModel], optional): 既存のモデルインスタンス
    """
    print(f"類似画像を検索中: {query_image_path}")
    
    if not os.path.exists(query_image_path):
        print(f"クエリ画像が見つかりません: {query_image_path}")
        return
        
    db_manager = DatabaseManager(
        host=config['postgres-host'],
        port=config['postgres-port'],
        user=config['postgres-user'],
        password=config['postgres-password'],
        database=config['postgres-database']
    )
    
    try:
        db_manager.connect()
        
        # モデルが渡されていない場合のみ新規作成
        if model is None:
            model = ImageEmbeddingModel(config['model-name'], config['device'])
        
        query_embedding = model.encode_image(query_image_path)
        
        results = db_manager.search_similar_images(query_embedding, limit=10)
        
        if not results:
            print("類似画像が見つかりませんでした")
            return
            
        print(f"{len(results)}個の類似画像を発見しました")
        for i, result in enumerate(results[:5], 1):
            similarity = result.get('cosine_similarity', 0)
            print(f"ランキング {i}: {result['file_name']} (類似度: {similarity:.4f})")
        
        output_dir = create_output_directory(config['output-directory'])
        
        best_match = results[0]
        best_match_path = best_match['file_path']
        
        save_search_results(
            output_dir=output_dir,
            config=config,
            results=results,
            query_image_path=query_image_path,
            best_match_path=best_match_path
        )
        
    except Exception as e:
        print(f"検索エラー: {e}")
        raise
    finally:
        db_manager.close()


def search_similar_image(query_image_path: str, config: Dict[str, Any]) -> None:
    """指定された画像と類似する画像をデータベースから検索します。
    
    クエリ画像に対して前処理を行い、埋め込みベクトルを生成して
    データベース内の全ての画像との類似度を計算します。
    結果は指定されたディレクトリに保存されます。
    
    Args:
        query_image_path (str): 検索対象画像のパス
        config (Dict[str, Any]): システム設定の辞書
    """
    print(f"類似画像を検索中: {query_image_path}")
    
    if not os.path.exists(query_image_path):
        print(f"クエリ画像が見つかりません: {query_image_path}")
        return
        
    db_manager = DatabaseManager(
        host=config['postgres-host'],
        port=config['postgres-port'],
        user=config['postgres-user'],
        password=config['postgres-password'],
        database=config['postgres-database']
    )
    
    try:
        db_manager.connect()
        
        model = ImageEmbeddingModel(config['model-name'], config['device'])
        
        query_embedding = model.encode_image(query_image_path)
        
        results = db_manager.search_similar_images(query_embedding, limit=10)
        
        if not results:
            print("類似画像が見つかりませんでした")
            return
            
        print(f"{len(results)}個の類似画像を発見しました")
        for i, result in enumerate(results[:5], 1):
            similarity = result.get('cosine_similarity', 0)
            print(f"ランキング {i}: {result['file_name']} (類似度: {similarity:.4f})")
        
        output_dir = create_output_directory(config['output-directory'])
        
        best_match = results[0]
        best_match_path = best_match['file_path']
        
        save_search_results(
            output_dir=output_dir,
            config=config,
            results=results,
            query_image_path=query_image_path,
            best_match_path=best_match_path
        )
        
    except Exception as e:
        print(f"検索エラー: {e}")
        raise
    finally:
        db_manager.close()


def interactive_search(config: Dict[str, Any]) -> None:
    """インタラクティブな画像検索モードを実行します。
    
    ユーザーからの入力を受け付け、連続して画像検索を実行できる
    モードです。'quit'、'exit'、'q'で終了します。
    初回のみモデルを読み込み、以降は同じモデルインスタンスを再利用します。
    
    Args:
        config (Dict[str, Any]): システム設定の辞書
    """
    print("\n=== インタラクティブ画像検索 ===")
    print("画像ファイルのパスを入力してください。終了する場合は 'quit' を入力してください。")
    
    # 最初に一度だけモデルを読み込み
    model = None
    
    while True:
        user_input = input("\n画像パスを入力: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("終了します...")
            break
            
        if not user_input:
            continue
            
        if not os.path.exists(user_input):
            print(f"ファイルが見つかりません: {user_input}")
            continue
            
        try:
            # 初回のみモデルを作成
            if model is None:
                model = ImageEmbeddingModel(config['model-name'], config['device'])
            
            # 参照画像処理なしで検索のみ実行
            search_similar_image_only(user_input, config, model)
        except Exception as e:
            print(f"検索中にエラーが発生しました: {e}")


def cleanup_database(config: Dict[str, Any]) -> None:
    """データベースをクリーンアップします。
    
    Args:
        config (Dict[str, Any]): データベース設定を含む辞書
    """
    try:
        db_manager = DatabaseManager(
            host=config['postgres-host'],
            port=config['postgres-port'],
            user=config['postgres-user'],
            password=config['postgres-password'],
            database=config['postgres-database']
        )
        
        db_manager.connect()
        cursor = db_manager.conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS image_embeddings')
        print("データベースをクリーンアップしました")
        db_manager.close()
    except Exception as e:
        print(f"データベースクリーンアップエラー: {e}")


# グローバル設定変数（シグナルハンドラー用）
_global_config = None


def signal_handler(signum, frame):
    """シグナルハンドラー（Ctrl+C等での中断処理）。"""
    print("\n\nプログラムが中断されました。データベースをクリーンアップしています...")
    if _global_config:
        cleanup_database(_global_config)
    sys.exit(0)


def main():
    """メイン関数。コマンドライン引数を処理して適切な関数を実行します。
    
    --process-only: 参照画像の処理のみ実行
    --search: 指定された画像で検索実行
    オプションなし: 参照画像処理 + インタラクティブ検索
    
    プログラム終了時（正常終了・異常終了問わず）にデータベースをクリアします。
    """
    global _global_config
    
    parser = argparse.ArgumentParser(description="コサイン類似度を使用した画像検索システム")
    parser.add_argument("--env", default="src/config.json", help="設定ファイルのパス")
    parser.add_argument("--process-only", action="store_true", help="参照画像の処理のみ実行")
    parser.add_argument("--search", type=str, help="指定されたファイルと類似した画像を検索")
    parser.add_argument("--no-cleanup", action="store_true", help="終了時のデータベースクリーンアップを無効化")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.env)
        _global_config = config
        
        # シグナルハンドラーを登録（Ctrl+C等）
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 正常終了時のクリーンアップを登録
        if not args.no_cleanup:
            atexit.register(lambda: cleanup_database(config))
        
        if args.process_only:
            process_reference_images(config)
        elif args.search:
            process_reference_images(config)
            search_similar_image(args.search, config)
        else:
            process_reference_images(config)
            interactive_search(config)
            
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みによりプログラムを終了します...")
        if _global_config and not args.no_cleanup:
            cleanup_database(_global_config)
        sys.exit(0)
    except Exception as e:
        print(f"エラー: {e}")
        if _global_config and not args.no_cleanup:
            cleanup_database(_global_config)
        sys.exit(1)


if __name__ == "__main__":
    main()
