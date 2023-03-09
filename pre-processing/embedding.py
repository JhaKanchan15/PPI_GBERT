# Install bio_embeddings using the command: pip install bio-embeddings[all]

from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
import numpy as np
import torch 

seq = 'MVTYDFGSDEMHD' # A protein sequence of length L

embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]
np_arr = protein_embd.cpu().detach().numpy()


