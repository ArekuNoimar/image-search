import os
import torch
import huggingface_hub
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)