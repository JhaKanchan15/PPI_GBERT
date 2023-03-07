

import os
import sys
import json
import pickle
from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder, Word2VecEmbedder
from Bio import SeqIO
import numpy as np
import torch 

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.cuda("cpu")
embedder = ProtTransBertBFDEmbedder()

count = 0

p_id = []
with open('Human/proteinList.txt') as f:
    for line in f:
        a = (line.strip()).split('\t')
        p_id.append(a)

seq = []
with open('Human/sequenceList.txt', encoding='utf8') as f:
    for line in f:
        a = line.strip()
        seq.append(a)

print(len(p_id), len(seq))

data = {}

for l, i in enumerate(p_id):
  count+= 1
  if l==len(seq):
    break 
    
  seq1 = seq[l]
  #print(seq1)
  if(len(seq1)> 7000):
    continue
  embedding = embedder.embed(seq1)
  #print('embedding = ', embedding)
  
  protein_embd = torch.tensor(embedding).sum(dim=0)#.mean(dim=0) 
  #print('len of protein embed = ',protein_embd.shape)# Vector with shape [1024]
  #protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L][1024]
  #np_arr = protein_embd.cpu().detach().numpy()
  np_arr = protein_embd.tolist()
  id1 = i[1]
  #print('np_arr = ',np_arr)
  print('len of np_arr = ', len(np_arr))
  
  #print(embedding.shape)
  print("Count = ",count," and file = ",i[0])
  data.update({id1:np_arr})
  #print(len(data))
  #print(data)
  #break


np.save('Human/Human_protTrans_dict.npy', data)


