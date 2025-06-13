"""  
Initialize files to store outputs and errors
"""

import os
import sys
from itertools import product

def init_dummy():
    """
    Direct output and error to standard output and error
    """
    global error, info
    error = sys.stderr
    info = sys.stdout

def init_files(sub_dir: str, init_info=True, init_cftfile=False, init_cft4=False):
    """
    Initialize files to store calculation result
    """
    os.makedirs(sub_dir, exist_ok=True)
    if init_info:
        global error, info
        # store calculation informations and errors
        error = open('{}/error.txt'.format(sub_dir),  mode='w', buffering=1)
        # tensor norm
        info  = open('{}/Tnorm.txt'.format(sub_dir),  mode='w', buffering=1)
    # CFT scaling dimensions
    if init_cftfile:
        w_list    = [2, 4] if init_cft4 else [2]
        type_list = ['v', 'h']
        bc_list   = ['p', 'a']
        par_list  = [0, 1]
        cftFilesNames = dict((
            key, '{}.txt'.format(key)
        ) for key in [
            'd{}{}-{}{}'.format(w, mattype, bc, vpar)
            for w, mattype, bc, vpar in product(
                w_list, type_list, bc_list, par_list
            )
        ])
        global cftFiles
        cftFiles = dict(
            (key, open('{}/{}'.format(sub_dir, filename), mode='w', buffering=1))
            for key, filename in cftFilesNames.items()
        )

def init_radfiles(sub_dir: str):
    """
    Initialize files to store radius result
    """
    # os.makedirs(sub_dir, exist_ok=True)
    global radFiles
    radFiles = dict(
        (key, open('{}/rad_{}.txt'.format(sub_dir, key), mode='w', buffering=1))
        for key in ['p0','a0']
    )

def closeall():
    """
    Close all files
    """
    if not (error is sys.__stderr__): error.close()
    if not (info is sys.__stdout__): info.close()
    try:
        for f in cftFiles.values(): f.close()
    except: 
        pass
    try:
        for f in radFiles.values(): f.close()
    except: 
        pass
