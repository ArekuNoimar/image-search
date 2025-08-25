# Image Search

## 概要

このリポジトリはコサイン疑似度を利用した画像検索システムです

システムの主なフローは下記の通りです

- 任意ディレクトリに存在する画像を フォーマット変更 & リサイズ & ベクトル化 & PostgreSQLへ登録
- gradioのGUIから、入力として1枚の画像をベクトル化し、PostgreSQL登録済ベクトルとコサイン疑似度検索を実施
- 検索結果のうち最も疑似度の高い画像を特定ディレクトリに出力

## リポジトリの構成

```bash
image-search/
├── README.md
├── apple.jpg                  # サンプルの入力データ
├── docker-compose.yml         # PostgreSQL
├── pyproject.toml
├── src
│   ├── config.json           # 変数設定ファイル
│   ├── db.py                 # DB関連のクラス・関数用のファイル
│   ├── images                # 参照画像保存用のディレクトリ
│   │   ├── apple.jpg
│   │   ├── banana.jpg
│   │   ├── grapes.jpg
│   │   └── strawberry.jpg
│   ├── main.py               # 主となる実行ファイル
│   ├── model.py              # モデル関連クラス・関数用のファイル
│   └── output                # 出力先ディレクトリ
└── uv.lock
```

## 初期設定

```bash
# ディレクトリ移動
cd image-search

# python 3.12.3の仮想環境を作成する
uv venv --python 3.12.3

# 仮想環境の有効化
source .venv/bin/activate

# 環境同期
uv sync
```

## 実行方法

環境変数をconfig.jsonから取得

```bash
# ディレクトリ移動
cd image-search

# 設定ファイルを--envオプションで指定
uv run src/main.py --env src/config.json
```

## 設定ファイル

デフォルトは"config.json"で、必要であれば "--env"オプションで指定

```json
{
 "source-directory":"src/images",    // 参照画像ディレクトリ
 "output-directory":"src/output",    // 出力先ディレクトリ
 "device":"cuda",                    // モデルの配置デバイス(cpu, cuda)
 "model-name":"jinaai/jina-clip-v2", // モデル名
 "dimension":1024,                   // 次元数
 "resize-width":960,                 // リサイズ時の画像横幅
 "resize-height":540                 // リサイズ時の画像縦幅
}
```

## 出力

変数設定ファイル(config.json)で指定した"output-directory"に保存する内容  

```bash
yyyymmdd-hhmmss
├── ○○○○.json   # 変数設定ファイル  
├── ○○○○.jpg    # 最近似画像(jpg, png等)  
├── ○○○○.jsonl  # 検索結果(一致率, 検索ファイル数, 検索ファイル名)
```

## ライブラリ

Pillow  
transformers  
huggingface_hub  
psycopg2  
torch  
tqdm

## 挙動

① jsonより変数取得
② "source-directory"で指定したディレクトリの.jpeg, png, webpをjpgに変換(※処理中はtqdmでプログレスバーを表示)  
③ "resize-width", "resize-height"で指定したサイズにリサイズ(※処理中はtqdmでプログレスバーを表示)  
④ "postgres-host", "postgres-port", "postgres-user", "postgres-password", "postgres-database"で指定したdbに接続  
⑤ テーブルの存在確認を実施した後、テーブルが存在していなければdbでテーブルを作成し、登録日付, ファイルパス, ファイル名, ファイルパス+ファイル名のハッシュ値, ベクトルデータ, 格納するためのカラムを作成  
⑥ リサイズ済み画像のファイル名からハッシュ値を生成、画像はclipモデルでEmbeddingしてベクトルデータを保持(※処理中はtqdmでプログレスバーを表示)  
⑦ dbに登録時に既存データのハッシュ値と登録対象データのハッシュ値を比較し、重複チェックを実施して登録(重複は破棄)(※処理中はtqdmでプログレスバーを表示)  
⑨ ユーザーが画像を入力するまで待機し、入力されたら前処理を実施する(jpg変換, リサイズ, Embeddingのみ)  
⑩ コサイン疑似度を利用してdb登録済データのEmbeddingベクトルデータと入力データのEmbeddingベクトルデータを比較し、ランキング化  
⑪ 検索結果1位のデータを取得し、保存したファイルパス+ファイル名で検索結果1位のデータをcopyし、"output-directory"で指定したディレクトリにyyyymmdd-hhmmss形式のディレクトリを作成し、変数設定ファイル, 検索結果とともに画像をデータを保存する  