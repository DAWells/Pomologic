"""
There are so many apples in this dataset.
Filter to the main, non-apple fruits.
"""

import os
import shutil
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

def batch_embeddings(model, preprocessed_image_list):
    embedding_list = [create_embedding(model, img) for img in preprocessed_image_list]
    return embedding_list

details = pd.read_csv("data/interim/tidy_details.csv")
details.fruit = pd.Categorical(details.fruit)

#top non-apple fruits
nafs = details.groupby("fruit").agg('size').sort_values()[-10:-1].index
naf_details = details[details.fruit.isin(nafs)]
naf_details.to_csv("data/interim/tidy_nonapple_details.csv", index=False)

# Create a metadata file for huggingface dataloader
metadata = pd.concat([naf_details.pom_id+".jpg", naf_details.pom_id], axis=1)
metadata.columns = ["file_name", "pom_id"]

# Make new non-apple folder
os.makedirs("data/external/nonapples")
metadata.to_csv("data/external/nonapples/metadata.csv", index=False)

# Copy top non apple thumbnails
metadata.file_name.apply(lambda f: shutil.copy(f"data/external/thumbnails/{f}", f"data/external/nonapples/{f}"))

# Make embedding
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

image_dataset = load_dataset("imagefolder", data_dir="data/external/nonapples")
# Remove train/test segmentation
image_dataset = image_dataset['train']
# Convert images to RGB if they are not already
image_dataset = image_dataset.map(lambda row: {"image": row['image'].convert("RGB")})
# Process images using HuggingFace processor
image_dataset = image_dataset.map(lambda row: {"preprocessed_image": image_processor(images=row['image'], return_tensors="pt")})
# Set dataset format to PyTorch
image_dataset = image_dataset.with_format("pt")

# Process all at once, too large for laptop
# image_dataset = image_dataset.map(lambda img: {"embedding": create_embedding(model, img["preprocessed_image"])})
# Processes in batches
image_dataset = image_dataset.map(lambda img: {"embedding": batch_embeddings(model, img["preprocessed_image"])}, batch_size=32, batched=True)

embeddings = pd.concat(
    [
        pd.DataFrame(image_dataset['pom_id']),
        pd.DataFrame(image_dataset['embedding'])
    ],
    axis=1
)

embeddings.to_csv("data/interim/nonapple_embeddings.csv", index=False, header=False)
