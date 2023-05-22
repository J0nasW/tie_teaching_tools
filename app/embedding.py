from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

from sklearn.manifold import TSNE

import plotly.express as px

import torch
import pandas as pd
import streamlit as st

import numpy as np

def create_embedding(text, checkpoint, max_length=512):
    try:
        # torch.cuda.empty_cache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            device = "mps"
        print(f"Torch is using device: {device} for embedding generation.")

        embedding = TransformerDocumentEmbeddings(checkpoint)
        
        if len(text) > max_length:
            text = text[:max_length]
        sentence = Sentence(text)
        embedding.embed(sentence)
        
        return sentence.embedding.cpu().detach().numpy()
    except Exception as e:
        print(e)
        return None

def visualize_embedding_space(df):
    try:
        # torch.cuda.empty_cache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            device = "mps"
        print(f"Torch is using device: {device} for embedding visualization.")
        
        embeddings = np.array(df["Embedding Vector"].tolist())
        
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embeddings shape: {embeddings[0].shape}")
        
        embeddings_tsne = TSNE(
                n_components=2,
                perplexity=25,
                random_state=42
        ).fit_transform(
            embeddings.reshape(
                len(embeddings), -1
            )
        )
        
        # embeddings_tsne = embeddings_tsne.reshape(-1, 2)
        
        print(f"Embeddings TSNE shape: {embeddings_tsne.shape}")
        
        df["embedding_tsne_x"] = embeddings_tsne[:, 0]
        df["embedding_tsne_y"] = embeddings_tsne[:, 1]
        
        fig = px.scatter(
            df,
            x = df["embedding_tsne_x"],
            y = df["embedding_tsne_y"],
            hover_data="Filename",
            title="Embeddings Visualization"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return fig, df
    except Exception as e:
        print(e)
        return None, df
    