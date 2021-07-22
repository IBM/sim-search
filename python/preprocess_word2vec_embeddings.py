# (c) Copyright IBM Corporation 2019, 2020, 2021

from __future__ import print_function
from _simsearch import embeddings

import os

def _get_mpi_local_rank():
    val = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
    if val is None:
        return "0"
    else:
        return val

local_rank = _get_mpi_local_rank()

# LOAD EMBEDDINGS
if local_rank == "0":
    embeddings.load_word2vec_embeddings('../Embeddings/GoogleNews-vectors-negative300.bin', 'vectors', 'vocabulary.txt')

