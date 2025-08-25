# Image Search

## 概要

このリポジトリはコサイン類似度を利用した画像検索システムです。

システムの主なフローは下記の通りです:

- 任意ディレクトリに存在する画像をフォーマット変換・リサイズ・ベクトル化してPostgreSQLへ登録
- PostgreSQL登録済みベクトルとコサイン類似度検索を実行
- 検索結果のうち最も類似度の高い画像を特定ディレクトリに出力

## リポジトリの構成

```bash
image-search/
├── README.md
├── apple.jpg                     # サンプルの入力データ
├── docker-compose.yml            # PostgreSQL設定
├── pyproject.toml                # Python依存関係
├── demo.py                       # デモ実行ファイル
├── test_basic.py                 # 基本機能テスト
├── test_integration.py           # 統合テスト
├── test_interactive.py           # インタラクティブテスト
├── test_multiple_search.py       # 複数検索最適化テスト
├── test_signal_handling.py       # シグナルハンドリングテスト
├── src/
│   ├── config.json               # 設定ファイル
│   ├── db.py                     # データベース操作クラス・関数
│   ├── images/                   # 元画像保存ディレクトリ
│   │   ├── apple-2.jpg
│   │   ├── banana.jpg
│   │   ├── grapes.jpg
│   │   └── strawberry.jpg
│   ├── processed/                # 処理済み画像ディレクトリ（自動生成）
│   ├── main.py                   # メイン実行ファイル
│   ├── model.py                  # 画像処理・モデル関連クラス・関数
│   └── output/                   # 検索結果出力ディレクトリ
└── uv.lock
```

## 前提条件

- Python 3.12+
- Docker / Docker Compose
- uv (Python パッケージマネージャ)

## 初期設定

### 1. PostgreSQLサーバーの起動

```bash
# PostgreSQLコンテナを起動
docker compose up -d
```

### 2. Python環境の設定

```bash
# ディレクトリ移動
cd image-search

# Python 3.12の仮想環境を作成
uv venv --python 3.12

# 仮想環境の有効化
source .venv/bin/activate

# 依存関係をインストール
uv sync
```

## 実行方法

### 基本的な使用方法

```bash
# 基本実行（参照画像の処理 + インタラクティブ検索）
uv run src/main.py --env src/config.json

# 特定画像での検索実行
uv run src/main.py --env src/config.json --search apple.jpg

# 参照画像の処理のみ実行
uv run src/main.py --env src/config.json --process-only

# データベースクリーンアップを無効にして実行
uv run src/main.py --env src/config.json --no-cleanup
```

### テスト実行

```bash
# 基本機能テスト
python test_basic.py

# 統合テスト
python test_integration.py

# 複数検索最適化テスト
python test_multiple_search.py

# インタラクティブテスト
python test_interactive.py

# シグナルハンドリングテスト
python test_signal_handling.py

# デモ実行
python demo.py
```

## 設定ファイル

設定は `src/config.json` で管理されます。`--env` オプションで別ファイルを指定可能。

```json
{
  "source-directory": "src/images",          // 元画像ディレクトリ
  "processed-directory": "src/processed",    // 処理済み画像ディレクトリ
  "output-directory": "src/output",          // 出力先ディレクトリ  
  "device": "cuda",                          // 計算デバイス（cpu/cuda）
  "model-name": "jinaai/jina-clip-v2",       // 使用モデル名
  "dimension": 1024,                         // 埋め込みベクトル次元数
  "resize-width": 960,                       // リサイズ時の画像横幅
  "resize-height": 540,                      // リサイズ時の画像縦幅
  "postgres-host": "localhost",              // PostgreSQLホスト
  "postgres-port": 15432,                    // PostgreSQLポート
  "postgres-user": "postgres",               // PostgreSQLユーザー名
  "postgres-password": "postgres",           // PostgreSQLパスワード
  "postgres-database": "postgres"            // PostgreSQLデータベース名
}
```

## 出力形式

検索実行後、`output-directory` に以下の形式でファイルが保存されます:

```bash
yyyymmdd-hhmmss/
├── [検索画像名].json   # 使用した設定ファイル
├── [検索画像名].jpg    # 最も類似した画像
└── [検索画像名].jsonl  # 検索結果詳細（ランキング・類似度・ファイル情報）
```

### 検索結果ファイル例（.jsonl）

```json
{"rank": 1, "similarity": 0.9823, "file_name": "apple_processed.jpg", "file_path": "src/images/apple.jpg"}
{"rank": 2, "similarity": 0.8456, "file_name": "strawberry_processed.jpg", "file_path": "src/images/strawberry.jpg"}
{"rank": 3, "similarity": 0.7234, "file_name": "grapes_processed.jpg", "file_path": "src/images/grapes.jpg"}
```

## 依存ライブラリ

- **Pillow**: 画像処理
- **transformers**: CLIP モデル読み込み
- **huggingface_hub**: モデルダウンロード
- **psycopg2**: PostgreSQL接続
- **torch**: 深層学習フレームワーク
- **torchvision**: 画像処理用PyTorchライブラリ
- **tqdm**: プログレスバー表示
- **numpy**: 数値計算
- **einops**: テンソル操作
- **timm**: 画像モデルライブラリ

## システム動作フロー

### 初回実行時（参照画像処理）
1. **設定読み込み**: config.jsonから各種設定を取得
2. **画像前処理**: 
   - `source-directory` 内の.jpeg/.png/.webp画像を検出
   - JPG形式に変換し `processed-directory` に保存
   - 指定サイズにリサイズ
3. **データベース接続**: PostgreSQLに接続し、必要に応じてテーブル作成
4. **埋め込みベクトル生成**: 
   - 処理済み画像のMD5ハッシュ値を生成
   - CLIP モデルで画像を埋め込みベクトルに変換
   - 重複チェック後、データベースに保存

### 検索実行時
5. **検索処理**:
   - 検索対象画像を前処理・ベクトル化
   - データベース内の全ベクトルとコサイン類似度を計算
   - 類似度順にランキング
6. **結果出力**: 
   - タイムスタンプ付きディレクトリを作成
   - 最も類似した画像・設定ファイル・検索結果詳細を保存

### インタラクティブ検索時（最適化）
7. **連続検索最適化**:
   - モデルを1回のみ読み込み、メモリに保持
   - 2回目以降の検索で参照画像処理をスキップ
   - 高速な検索処理を実現（約44%の高速化）

### 終了時
8. **自動クリーンアップ**:
   - 正常終了時: データベースを自動クリーンアップ
   - 異常終了時: シグナルハンドラーによるクリーンアップ
   - `--no-cleanup` オプションでクリーンアップを無効化可能

## 新機能・最適化

### 処理済み画像の分離保存
- **元画像保護**: `source-directory` の元画像ファイルはそのまま保持
- **処理済み分離**: JPG変換・リサイズ済み画像は `processed-directory` に自動保存
- **ディレクトリ自動作成**: 処理済みディレクトリが存在しない場合は自動作成

### 複数検索最適化
- **モデル再利用**: インタラクティブモードで同一モデルインスタンスを再利用
- **高速化実現**: 2回目以降の検索で約44%の高速化を達成
- **参照画像処理スキップ**: 連続検索時は参照画像の前処理を省略

### 自動データベースクリーンアップ
- **正常終了時**: プログラム終了時に自動でデータベースをクリーンアップ
- **異常終了対応**: Ctrl+C等での強制終了時もクリーンアップを実行
- **オプション制御**: `--no-cleanup` フラグでクリーンアップを無効化可能

### GPU対応・パフォーマンス最適化
- **CUDA自動検出**: GPU利用可能時は自動でCUDA使用
- **BFloat16対応**: GPU使用時のテンソル型エラーを修正
- **高速処理**: RTX 4090での最適化済み動作確認

## 注意事項

- CLIP モデルの初回実行時は、モデルファイルのダウンロードに時間がかかります
- CPUでの実行は処理時間が長くなります。可能であればCUDA対応GPUの使用を推奨
- PostgreSQLサーバーが起動していることを確認してから実行してください

## トラブルシューティング

### データベース接続エラー
```bash
# PostgreSQLコンテナの状態確認
docker compose ps

# コンテナ再起動
docker compose restart

# データベースを手動でクリーンアップ
uv run src/main.py --env src/config.json --process-only
```

### GPU関連エラー
```bash
# CUDA利用可能性を確認
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# GPU使用時のメモリエラーが発生した場合はCPUモードに変更
# config.json で "device": "cpu" に設定
```

### 処理済みディレクトリ関連エラー
```bash
# 処理済みディレクトリを手動作成
mkdir -p src/processed

# 処理済みファイルをクリア
rm -rf src/processed/*
```

### モデルダウンロードエラー
インターネット接続とHugging Face へのアクセスを確認してください。

### 依存関係エラー
```bash
# 依存関係を再インストール
uv sync --reinstall

# 特定のパッケージが不足している場合
pip install einops timm torchvision
```  