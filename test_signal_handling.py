#!/usr/bin/env python3

"""シグナルハンドリング（強制終了時のDBクリーンアップ）テスト用スクリプト。

Ctrl+Cや強制終了時にDBがクリーンアップされることを確認するテストスクリプトです。
"""

import sys
import os
import time
import signal
import subprocess
sys.path.append('src')

from main import load_config
from db import DatabaseManager

def test_signal_handling():
    """シグナルハンドリングとDBクリーンアップをテストします。"""
    print("=== シグナルハンドリング・DBクリーンアップテスト ===\n")
    
    try:
        # 設定を読み込み
        config = load_config('src/config.json')
        
        # 現在のDBの状態を確認
        db_manager = DatabaseManager(
            host=config['postgres-host'],
            port=config['postgres-port'],
            user=config['postgres-user'],
            password=config['postgres-password'],
            database=config['postgres-database']
        )
        
        print("1. 現在のDB状態を確認中...")
        db_manager.connect()
        
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'image_embeddings'")
        table_exists_before = cursor.fetchone()[0] > 0
        
        if table_exists_before:
            cursor.execute("SELECT COUNT(*) FROM image_embeddings")
            record_count = cursor.fetchone()[0]
            print(f"✅ image_embeddings テーブル存在: {record_count}件のレコード")
        else:
            print("⚠️  image_embeddings テーブルが存在しません")
        
        db_manager.close()
        
        print("\n2. 強制終了テスト用のプロセスを起動...")
        print("   (5秒後に自動でCtrl+Cシグナルを送信します)")
        
        # バックグラウンドでインタラクティブモードを起動
        process = subprocess.Popen([
            'uv', 'run', 'src/main.py', '--env', 'src/config.json'
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, 'VIRTUAL_ENV': '.venv'}
        )
        
        # 5秒待機してからSIGINTを送信
        time.sleep(5)
        print("   Ctrl+Cシグナルを送信中...")
        process.send_signal(signal.SIGINT)
        
        # プロセス終了を待機
        stdout, stderr = process.communicate(timeout=10)
        
        print(f"   プロセス終了コード: {process.returncode}")
        
        # 終了後のDB状態を確認
        print("\n3. 終了後のDB状態を確認中...")
        db_manager.connect()
        
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'image_embeddings'")
        table_exists_after = cursor.fetchone()[0] > 0
        
        if table_exists_after:
            print("❌ テーブルがクリーンアップされていません")
            return False
        else:
            print("✅ テーブルが正常にクリーンアップされました")
            
        db_manager.close()
        
        print("\n🎉 シグナルハンドリング・DBクリーンアップテスト成功!")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ プロセスがタイムアウトしました")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    test_signal_handling()