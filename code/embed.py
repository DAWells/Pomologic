"""
Generate embeddings for images following
https://www.fuzzylabs.ai/blog-post/hugging-face-in-space
"""

import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTMAEModel

def create_embedding(model, preprocessed_image) -> torch.Tensor:
    """Passes a preprocessed image through a pretrained embedding model.

    Args:
        model (PreTrainedModel): Pretrained HuggingFace PyTorch embedding model.
        preprocessed_image (torch.Tensor): Preprocessed image as a PyTorch Tensor

    Returns:
        torch.Tensor: Embedding vector shape (1, 768) as a Tensor
    """
    embedding = model(**preprocessed_image).last_hidden_state[:, 0]
    return np.squeeze(embedding)

# Create a metadata file for huggingface dataloader
details = pd.read_csv("data/external/details.csv", names=["pom_id", "description", "author", "date", "image_src"])
metadata = pd.concat([details.pom_id+".jpg", details.pom_id], axis=1)
metadata.columns = ["file_name", "pom_id"]
metadata.to_csv("data/external/thumbnails/metadata.csv", index=False)

model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

image_dataset = load_dataset("imagefolder", data_dir="data/external/thumbnails")
# Remove train/test segmentation
image_dataset = image_dataset['train']
# Convert images to RGB if they are not already
image_dataset = image_dataset.map(lambda row: {"image": row['image'].convert("RGB")})
# Process images using HuggingFace processor
image_dataset = image_dataset.map(lambda row: {"preprocessed_image": image_processor(images=row['image'], return_tensors="pt")})
# Set dataset format to PyTorch
image_dataset = image_dataset.with_format("pt")

image_dataset = image_dataset.map(lambda img: {"embedding": create_embedding(model, img["preprocessed_image"])})

embeddings = pd.concat(
    [
        pd.DataFrame(image_dataset['pom_id']),
        pd.DataFrame(image_dataset['embedding'])
    ],
    axis=1
)

embeddings.to_csv("data/interim/embeddings.csv", index=False, header=False)
