from GIO import GIOKL
import torch
import os
import numpy as np
import jax.numpy as jnp
from loguru import logger

DATA_PATH='data/'

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
        syn_emb_df = gio_kl.spark.createDataFrame(data[(i, syn_emb[i, :].squeeze()) for i in range(syn_emb.shape[0])] , schema=['id', 'features'])
        syn_emb_df = gio_kl.spark.createDataFrame(data[(i, target_emb[i, :].squeeze()) for i in range(target_emb.shape[0])] , schema=['id', 'features'])
        model_train, model_X, transformed_train, transformed_X = gio_kl.quantize(syn_emb_df, target_emb_df)

        X = jnp.array(model_X.clusterCenters())
        train = jnp.array(model_train.clusterCenters())
        centroids_df = gio_kl.spark.createDataFrame(data=[(i, each.tolist()) for i, each in enumerate(model_train.clusterCenters())], schema=["id", "centroid"])
        logger.success(f'Successfully quantized data.')

        logger.info('Select data ...')
        W, kl_divs, _ = gio_kl.fit(train, X, max_iter=300, stopping_criterion='sequential_increase_tolerance', v_init='jump')
        W = W[20:] # Remove the uniform start
        logger.success('Successfully selected data.')

        logger.info('Explode data ...')
        full_selections_df = gio_kl.explode(W, transformed_train, centroids_df)