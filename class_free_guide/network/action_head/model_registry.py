from .utils.utils import *
from .cross_attention_dit import DiT

MODEL_REGISTRY = {
    "MLP": CondCategorySpecificMLP,
    "Transformer": MultiEmbodimentActionEncoder,  # unimodal model
    "DiT": DiT,  # multimodal model
}
