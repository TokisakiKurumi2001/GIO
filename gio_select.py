from GIO import GIOKL
import torch
import os
import numpy as np
import jax.numpy as jnp
from loguru import logger
from sklearn.cluster import KMeans

DATA_PATH='data/'

def quantize(np_array, num_centroid=1500):
    model = MiniBatchKMeans(n_clusters=num_centroid, random_state=42, max_iter=20, verbose=1, n_init='auto')
    outputs = model.fit(np_array) # clustering
    centroid_array = outputs.cluster_centers_
    vec2cluster_id = outputs.labels_
    return centroid_array, vec2cluster_id

def explode(centroid_array, selected_centroid_array, vec2cluster_id):
    cluster_ids = []

    # Iterate through each sample in Tensor A
    for sample in selected_centroid_array:
        # Find matching indices in Tensor B
        match_indices = np.where((centroid_array == sample).all(axis=1))[0]

        # Add found indices to cluster_ids
        cluster_ids.extend(match_indices.tolist())

    indicies = []
    for idx, cluster_id in enumerate(vec2cluster_id):
        if cluster_id in cluster_ids:
            indicies.append(idx)

    return indicies

if __name__ == "__main__":
    logger.info('Loading encoded data ...')
    syn_emb = torch.load(DATA_PATH + "syn.pt")
    logger.success(f'Successfully load syn.pt data.')
    
    if os.path.exists(DATA_PATH + 'select.pt'):
        pass
    else:
        logger.info('Loading target encoded data ...')
        target_emb = torch.load(DATA_PATH + "valid.pt")
        logger.success(f'Successfully load valid.pt data.')

        gio_kl = GIOKL.GIOKL(uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=4096)

        logger.info('Quantize data ...')
        syn_emb = syn_emb.view(-1, 4096)
        target_emb = target_emb.view(-1, 4096)
        syn_centroid, syn_mappings = quantize(syn_emb.numpy(), 5000)
        target_centroid, _ = quantize(target_emb.numpy(), 500)

        X = jnp.array(target_centroid)
        train = jnp.array(syn_centroid)
        logger.success(f'Successfully quantized data.')

        logger.info('Select data ...')
        W, kl_divs, _ = gio_kl.fit(train, X, max_iter=300, stop_criterion='increase', v_init='jump')
        W = W[20:] # Remove the uniform start
        logger.success('Successfully selected data.')

        logger.info('Explode data ...')
        W_np = np.array(W, copy=False)
        indices = explode(syn_centroid, W_np, syn_mappings)
        torch.save(torch.tensor(indices), DATA_PATH + 'select.pt')