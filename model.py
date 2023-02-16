import numpy as np
import pandas as pd
import pickle
import gdown
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Ghani-25/LF_enrich_sim", device='cuda')
url = "https://drive.google.com/uc?export=download&id=1-GEVcdInQ1QIaPsOYKJDoUcxmVpnhmFM"
output = "Embeddings_full"
gdown.download(url, output, quiet=False)
with open("./Embeddings_full", "rb") as fp:
  Embeddings = pickle.load(fp)

def enrichir(tab):
    #Compute cosine-similarities with all embeddings
    query = '. '.join(tab)
    query_embedd = model.encode(query)
    cosine_scores = util.pytorch_cos_sim(query_embedd, Embeddings)
    Similarities = torch.sort(cosine_scores,descending=True)
    print('Les taux de similarités sont :', Similarities)
    top_matches = torch.argsort(cosine_scores, dim=-1, descending=True).tolist()[0][0:20]
    print(top_matches)
    return top_matches
