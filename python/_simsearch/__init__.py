# (c) Copyright IBM Corporation 2019, 2020, 2021

import os
import sys

def _is_mpi_enabled():
    val = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
    if val is None:
        return False
    else:
        return True

using_mpi = _is_mpi_enabled()
if using_mpi:
    from _simsearch import libpywmdmpi as libpywmd
else:
    from _simsearch import libpywmd as libpywmd

