"""画像処理とCLIPモデルによる埋め込みベクトル生成を行うモジュール。

このモジュールは画像検索システムの画像処理とモデル操作を担当します。
主な機能として、画像のフォーマット変換、リサイズ、CLIP モデルによる
埋め込みベクトルの生成を提供します。
"""

import os
import torch
import numpy as np
from PIL import Image
import huggingface_hub
from transformers import AutoModel, AutoProcessor
from typing import List, Tuple
from tqdm import tqdm


class ImageEmbeddingModel:
    """CLIP モデルを使用した画像埋め込みベクトル生成クラス。
    
    指定されたCLIP モデルを読み込み、画像ファイルから埋め込みベクトルを
    生成する機能を提供します。CPUまたはGPU上でモデルを実行可能です。
    
    Attributes:
        model_name (str): 使用するモデル名
        device (str): 実行デバイス（'cpu'または'cuda'）
        model: 読み込まれたCLIP モデル
        processor: 画像前処理用プロセッサ
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """ImageEmbeddingModel を初期化します。
        
        Args:
            model_name (str): 使用するHugging Face モデル名
            device (str, optional): 実行デバイス。デフォルトは 'cpu'。
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        """指定されたモデルを読み込み、デバイスに配置します。
        
        Hugging Face からモデルとプロセッサをダウンロードし、
        指定されたデバイス（CPUまたはGPU）に配置します。
        """
        print(f"モデルを読み込み中: {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda").to(torch.float32)
            print("モデルをGPU（CUDA）に配置しました")
        else:
            self.model = self.model.to(torch.float32)
            print("モデルをCPUに配置しました")
            
    def encode_image(self, image_path: str) -> np.ndarray:
        """単一の画像ファイルを埋め込みベクトルに変換します。
        
        Args:
            image_path (str): 画像ファイルのパス
            
        Returns:
            np.ndarray: 画像の埋め込みベクトル（1次元配列）
        """
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda").to(torch.float32) if v.dtype.is_floating_point else v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            
        return embeddings.cpu().numpy().flatten()
        
    def encode_images_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """複数の画像ファイルをバッチで埋め込みベクトルに変換します。
        
        Args:
            image_paths (List[str]): 画像ファイルパスのリスト
            
        Returns:
            List[np.ndarray]: 埋め込みベクトルのリスト。エラー時はNoneが含まれます。
        """
        embeddings = []
        for image_path in tqdm(image_paths, desc="画像を埋め込みベクトルに変換中"):
            try:
                embedding = self.encode_image(image_path)
                embeddings.append(embedding)
            except Exception as e:
                print(f"画像の埋め込み変換エラー {image_path}: {e}")
                embeddings.append(None)
        return embeddings


def convert_to_jpg(image_path: str, output_path: str) -> bool:
    """画像ファイルをJPG形式に変換します。
    
    Args:
        image_path (str): 入力画像ファイルのパス
        output_path (str): 出力JPGファイルのパス
        
    Returns:
        bool: 変換が成功した場合True、失敗した場合False
    """
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"JPG変換エラー {image_path}: {e}")
        return False


def resize_image(image_path: str, output_path: str, width: int, height: int) -> bool:
    """画像を指定されたサイズにリサイズします。
    
    Args:
        image_path (str): 入力画像ファイルのパス
        output_path (str): 出力画像ファイルのパス
        width (int): リサイズ後の幅
        height (int): リサイズ後の高さ
        
    Returns:
        bool: リサイズが成功した場合True、失敗した場合False
    """
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"画像リサイズエラー {image_path}: {e}")
        return False


def process_images_in_directory(source_dir: str, processed_dir: str, resize_width: int, resize_height: int) -> List[str]:
    """ディレクトリ内の画像ファイルを処理します。
    
    指定されたソースディレクトリから対応する形式の画像ファイルを検索し、
    JPG形式への変換とリサイズを行って処理済みディレクトリに保存します。
    
    Args:
        source_dir (str): 処理対象ディレクトリのパス
        processed_dir (str): 処理済み画像を保存するディレクトリのパス
        resize_width (int): リサイズ後の幅
        resize_height (int): リサイズ後の高さ
        
    Returns:
        List[str]: 処理された画像ファイルのパスリスト
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    processed_images = []
    
    if not os.path.exists(source_dir):
        print(f"ソースディレクトリが存在しません: {source_dir}")
        return processed_images
        
    # 処理済みディレクトリを作成
    os.makedirs(processed_dir, exist_ok=True)
    print(f"処理済みディレクトリを準備しました: {processed_dir}")
        
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"{len(image_files)}個の画像ファイルを発見しました")
    
    for image_path in tqdm(image_files, desc="画像を処理中"):
        try:
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            temp_jpg_path = os.path.join(processed_dir, f"{name}_temp.jpg")
            final_path = os.path.join(processed_dir, f"{name}_processed.jpg")
            
            if convert_to_jpg(image_path, temp_jpg_path):
                if resize_image(temp_jpg_path, final_path, resize_width, resize_height):
                    processed_images.append(final_path)
                    if temp_jpg_path != final_path:
                        try:
                            os.remove(temp_jpg_path)
                        except:
                            pass
                            
        except Exception as e:
            print(f"画像処理エラー {image_path}: {e}")
            
    print(f"{len(processed_images)}個の画像を正常に処理しました")
    print(f"処理済み画像は以下に保存されました: {processed_dir}")
    return processed_images