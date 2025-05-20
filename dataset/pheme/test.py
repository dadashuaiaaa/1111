import torch 
from PIL import Image
import clip
import cn_clip.clip as clip
from clip import load_from_name, available_models
print("Available models:", available_models())  