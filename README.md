# Image Search

## 概要

参照画像を前処理して正規化 + CLIPモデルでベクトル化し、コサイン類似度を利用して画像検索を実施するシステムです

システムの主なフローは下記の通りです:

- 任意ディレクトリに存在する画像をフォーマット変換・リサイズ・ベクトル化してPostgreSQLへ登録
- 入力画像をベクトル化してPostgreSQL登録済みベクトルとコサイン類似度検索を実行
- 検索結果のうち最も類似度の高い画像を特定ディレクトリに出力

## リポジトリの構成

```bash
image-search/
├── README.md
├── sample-apple.jpg              # サンプルの入力データ
├── docker-compose.yml            # PostgreSQL設定
├── pyproject.toml                # Python依存関係
├── src/
│   ├── config.json               # 設定ファイル
│   ├── db.py                     # データベース操作クラス・関数
│   ├── images/                   # 元画像保存ディレクトリ
│   │   ├── apple.webp
│   │   ├── banana.png
│   │   ├── grapes.jpeg
│   │   └── strawberry.jpg
│   ├── processed/                # 処理済み画像ディレクトリ（自動生成）
│   ├── main.py                   # メイン実行ファイル
│   ├── model.py                  # 画像処理・モデル関連クラス・関数
│   └── output/                   # 検索結果出力ディレクトリ
└── uv.lock
```

## 前提条件

- Python 3x
- UV  
- Docker  
- Nvidia-Driver, Nvidia-Cuda-Toolkit, Nvidia-Container-Toolkit  

## 動作確認済環境

### 環境情報（表形式）

| 項目 | 値 |
|---|---|
| OS | Ubuntu 22.04.5 Desktop LTS |
| GPU | Nvidia GeForce RTX 4090 Laptop 16GB |
| UV | 0.8.4 |
| Python | 3.12.3 |
| Docker | 28.32 |
| Nvidia Driver | 575.51.03 |
| Nvidia CUDA Toolkit | 12.9 |
| Nvidia Container Toolkit | 1.18.0 |

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

# Python 3.12.3の仮想環境を作成
uv venv --python 3.12.3

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
uv run src/main.py --env src/config.json --search sample-apple.jpg

# 参照画像の処理のみ実行
uv run src/main.py --env src/config.json --process-only

# データベースクリーンアップを無効にして実行
uv run src/main.py --env src/config.json --no-cleanup
```

## 利用可能オプション

| オプション | 説明 | 型/値 | 既定値 | 例 |
|---|---|---|---|---|
| `--env` | 設定ファイルのパス | 文字列 | `src/config.json` | `--env src/config.json` |
| `--process-only` | 参照画像の処理のみ実行 | フラグ | `false` | `--process-only` |
| `--search` | 指定されたファイルと類似した画像を検索 | 文字列（画像パス） | なし | `--search sample-apple.jpg` |
| `--no-cleanup` | 終了時のデータベースクリーンアップを無効化 | フラグ | `false` | `--no-cleanup` |

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
  "resize-width": 1000,                       // リサイズ時の画像横幅
  "resize-height": 800,                      // リサイズ時の画像縦幅
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

## DBテーブル

PostgreSQL で作成されるテーブル: `image_embeddings`

| カラム名 | 型 | 制約 | 説明 |
|---|---|---|---|
| id | SERIAL | PRIMARY KEY | 自動採番 ID |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | レコード作成時刻 |
| file_path | TEXT | NOT NULL | 画像のフルパス（処理済み） |
| file_name | TEXT | NOT NULL | 画像ファイル名 |
| file_hash | TEXT | UNIQUE NOT NULL | 画像内容のMD5ハッシュ（重複排除に使用） |
| embedding | REAL[] | NOT NULL | 画像の埋め込みベクトル（例: 1024次元） |

例（1レコード）

| カラム名 | 例 |
|---|---|
| id | 1 |
| created_at | 2025-01-01 12:34:56 |
| file_path | src/processed/apple.jpg |
| file_name | apple.jpg |
| file_hash | 098f6bcd4621d373cade4e832627b4f6 |
| embedding | [0.0123, -0.4567, 0.0890, ..., 0.0345] |

補足

- `embedding` は Python 側から `list[float]` として挿入されます（`REAL[]`）。
- ベクトル次元は使用モデルに依存します（例: `config.json` の `dimension`: 1024）。
- 重複登録は `file_hash`（MD5）で防止しています。

## システム動作フロー

### 初回実行時（参照画像前処理）
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
   - 2回目以降の検索で参照画像前処理をスキップ

### 終了時
8. **自動クリーンアップ**:
   - 正常終了時: データベースを自動クリーンアップ
   - 異常終了時: シグナルハンドラーによるクリーンアップ
   - `--no-cleanup` オプションでクリーンアップを無効化可能

## 機能・最適化

### 処理済み画像の分離保存
- **元画像保護**: `source-directory` の元画像ファイルはそのまま保持
- **処理済み分離**: JPG変換・リサイズ済み画像は `processed-directory` に自動保存
- **ディレクトリ自動作成**: 処理済みディレクトリが存在しない場合は自動作成

### 複数検索最適化
- **モデル再利用**: インタラクティブモードで同一モデルインスタンスを再利用
- **参照画像前処理スキップ**: 連続検索時は参照画像の前処理を省略

### 自動データベースクリーンアップ
- **正常終了時**: プログラム終了時に自動でデータベースをクリーンアップ
- **異常終了対応**: Ctrl+C等での強制終了時もクリーンアップを実行
- **オプション制御**: `--no-cleanup` フラグでクリーンアップを無効化可能

### GPU対応
- **CUDA自動検出**: GPU利用可能時は自動でCUDA使用

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

## 環境構築

### UVインストール方法(Ubuntu)

```bash
# インストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# バージョン確認
uv --version
```

### Dockerインストール(Ubuntu)

[Docker](https://docs.docker.com/engine/install/ubuntu/)より参照

```bash
# GPGキーの追加
sudo apt-get update

sudo apt-get install ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

# apt設定
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

```bash
# パッケージのインストール
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```bash
# 権限変更
sudo groupadd docker

sudo usermod -aG docker $USER
```

## Nvidia-Driver, Nvidia-Cuda-Toolkitインストール(Ubuntu22系)(ver 12.9)

[CUDA Toolkit 12.9 Downloads](https://developer.nvidia.com/cuda-12-9-0-download-archive)より参照

```bash
# CUDA Toolkit インストール
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

```bash
# Nvidia-Driver　インストール
sudo apt-get install -y nvidia-open
```

### Nvidia-Container-Toolkitインストール

[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)より参照

```bash
# Nvidia Container Toolkitのgpgキー追加、aptリポジトリ設定
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
```

```bash
# パッケージのインストール
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

```bash
# サービス設定ファイル追加
sudo touch  /etc/docker/daemon.json

sudo vim /etc/dcker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```