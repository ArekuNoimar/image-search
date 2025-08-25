"""画像埋め込みベクトルのデータベース操作を行うモジュール。

このモジュールはPostgreSQLデータベースに対する画像埋め込みベクトルの
保存、検索、管理機能を提供します。コサイン類似度を使用した画像検索を
効率的に実行できます。
"""

import os
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Any, Optional


class DatabaseManager:
    """PostgreSQLデータベースでの画像埋め込みベクトル管理クラス。
    
    画像の埋め込みベクトルをデータベースに保存し、コサイン類似度を
    使用した画像検索機能を提供します。重複チェックや効率的な
    ベクトル検索をサポートします。
    
    Attributes:
        host (str): データベースホスト
        port (int): データベースポート
        user (str): データベースユーザー名
        password (str): データベースパスワード
        database (str): データベース名
        conn: データベース接続オブジェクト
    """
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """DatabaseManager を初期化します。
        
        Args:
            host (str): PostgreSQLサーバーのホスト名
            port (int): PostgreSQLサーバーのポート番号
            user (str): データベースユーザー名
            password (str): データベースパスワード
            database (str): 使用するデータベース名
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        
    def connect(self):
        """PostgreSQLデータベースに接続します。
        
        Raises:
            Exception: データベース接続に失敗した場合
        """
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.conn.autocommit = True
            print(f"PostgreSQLデータベースに接続しました {self.host}:{self.port}")
        except Exception as e:
            print(f"データベース接続エラー: {e}")
            raise
            
    def create_table(self):
        """画像埋め込みベクトル用のテーブルを作成します。
        
        既にテーブルが存在する場合は何も行いません。
        
        Raises:
            Exception: テーブル作成に失敗した場合
        """
        if not self.conn:
            raise Exception("データベース接続がありません")
            
        cursor = self.conn.cursor()
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                embedding REAL[] NOT NULL
            );
            """
            cursor.execute(create_table_sql)
            print("テーブル 'image_embeddings' を作成または確認しました")
        except Exception as e:
            print(f"テーブル作成エラー: {e}")
            raise
        finally:
            cursor.close()
            
    def hash_exists(self, file_hash: str) -> bool:
        """指定されたファイルハッシュがデータベースに存在するかチェックします。
        
        Args:
            file_hash (str): チェック対象のファイルハッシュ
            
        Returns:
            bool: ハッシュが存在する場合True、存在しない場合False
        """
        if not self.conn:
            raise Exception("データベース接続がありません")
            
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM image_embeddings WHERE file_hash = %s", (file_hash,))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            print(f"ハッシュ存在確認エラー: {e}")
            return False
        finally:
            cursor.close()
            
    def insert_embedding(self, file_path: str, file_name: str, file_hash: str, embedding: np.ndarray):
        """画像の埋め込みベクトルをデータベースに挿入します。
        
        Args:
            file_path (str): 画像ファイルのパス
            file_name (str): 画像ファイル名
            file_hash (str): 画像ファイルのハッシュ値
            embedding (np.ndarray): 画像の埋め込みベクトル
            
        Returns:
            bool: 挿入が成功した場合True、重複またはエラーの場合False
        """
        if not self.conn:
            raise Exception("データベース接続がありません")
            
        if self.hash_exists(file_hash):
            print(f"重複ファイルをスキップしました: {file_name}")
            return False
            
        cursor = self.conn.cursor()
        try:
            insert_sql = """
            INSERT INTO image_embeddings (file_path, file_name, file_hash, embedding)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_sql, (file_path, file_name, file_hash, embedding.tolist()))
            print(f"埋め込みベクトルを挿入しました: {file_name}")
            return True
        except Exception as e:
            print(f"埋め込みベクトル挿入エラー: {e}")
            return False
        finally:
            cursor.close()
            
    def search_similar_images(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """クエリ画像と類似する画像をコサイン類似度で検索します。
        
        Args:
            query_embedding (np.ndarray): 検索対象の埋め込みベクトル
            limit (int, optional): 返す結果数の上限。デフォルトは10。
            
        Returns:
            List[Dict[str, Any]]: 類似度順にソートされた検索結果のリスト
        """
        if not self.conn:
            raise Exception("データベース接続がありません")
            
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            all_embeddings = self.get_all_embeddings()
            if not all_embeddings:
                return []
                
            similarities = []
            for row in all_embeddings:
                stored_embedding = np.array(row['embedding'])
                similarity = cosine_similarity(query_embedding, stored_embedding)
                similarities.append({
                    'file_path': row['file_path'],
                    'file_name': row['file_name'],
                    'file_hash': row['file_hash'],
                    'embedding': row['embedding'],
                    'cosine_similarity': similarity
                })
            
            similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            print(f"類似画像検索エラー: {e}")
            return []
        finally:
            cursor.close()
            
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """データベースに保存されている全ての埋め込みベクトルを取得します。
        
        Returns:
            List[Dict[str, Any]]: 全ての埋め込みベクトルデータのリスト
        """
        if not self.conn:
            raise Exception("データベース接続がありません")
            
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT * FROM image_embeddings")
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"全埋め込みベクトル取得エラー: {e}")
            return []
        finally:
            cursor.close()
            
    def close(self):
        """データベース接続を閉じます。"""
        if self.conn:
            self.conn.close()
            print("データベース接続を閉じました")


def generate_file_hash(file_path: str) -> str:
    """ファイルのMD5ハッシュ値を生成します。
    
    ファイルを8192バイトずつ読み込んでMD5ハッシュを計算し、
    大きなファイルでもメモリ効率的に処理します。
    
    Args:
        file_path (str): ハッシュを計算するファイルのパス
        
    Returns:
        str: ファイルのMD5ハッシュ値（16進文字列）
    """
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """2つのベクトル間のコサイン類似度を計算します。
    
    Args:
        a (np.ndarray): 第1のベクトル
        b (np.ndarray): 第2のベクトル
        
    Returns:
        float: コサイン類似度（-1から1の範囲）
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))