"""
Set parameters for 
tensor network renormalization
"""

import numpy as np

def get_beta_tV(g):
    """
    Set beta for t-V model (when mu = -V)
        g = arccos(V / 2t) / pi
    Function
        β(g) = 2 g / sin(π g)
    """
    gamma = g * np.pi
    if np.pi * np.sin(gamma) < 0.01:
        """ 
            when g -> 0
            lim_{g->0} 2 g / sin(π g) = 1 / π
        """
        return 2.0 / np.pi
    else:
        return 2 * gamma / (np.pi * np.sin(gamma))

def get_beta_tVmu(mu):
    """
        Set beta for t-V model (currently support V = 0 (g = 0.5) only)
    """
    return 1 / np.cos(0.54 * mu)

def set_paramRG(model: str, additions=None):
    """
    Set default model parameters for TNR

    Parameters
    ----------
    model : str
        Statistical model
    additions : dict or NoneType
        Other parameters obtained from command line / input script
    """
    assert isinstance(additions, dict) or additions is None
    # ---- common parameters for TNR ----
    param = {'model': model, 'maxRGstep': 21, 
            'shrink1by1': False, 'verticalStep': 6,
            'stepCFT': 2, 'nev': 200, 'forceC': 0,
            '3body': False, '3bodyAB': False, 
            'mpo': False, 'h3to1': False, 'h2to1': False}
    # combine param and other_param
    param.update(additions)
    
    # ---- parameters for specific models ----
    # t-V model (t = 1)
    if 'tV' in model:
        if 't' not in param:
            param['t'] = 1
        # convert input g to V
        param['V'] = np.around(2 * np.cos(param['g'] * np.pi), decimals=15)
        # determine beta if not given
        if 'beta' not in param:
            if 'mu' in param:
                assert param['g'] == 0.5
                param['beta'] = get_beta_tVmu(param['mu'])
            else:
                assert 'mu' not in param
                param['beta'] = get_beta_tV(param['g'])
        # determine mu if not given
        if 'mu' not in param:
            param['mu'] = param['V']
    
    elif model == 'kitaev':
        param['t'] = 1
        # currently consider the symmetry point only
        # or return tV model (D = 0)
        if 'D' not in param:
            param['D'] = param['t']
        # determine V, mu if not given
        # known points of phase transition:
        # V = 0, mu = 2
        # V = 1/4, mu = 0
        # determine beta if not given
        if 'beta' not in param:
            if 'mu' in param:
                assert param['g'] == 0.5
                param['beta'] = get_beta_tVmu(param['mu'])
            else:
                assert 'mu' not in param
                param['beta'] = get_beta_tV(param['g'])

    # t-J model default values
    elif model == 'tJ':
        if 't' not in param:
            param['t'] = 1
        if 'beta' not in param:
            param['beta'] = 1
        if 'mu' not in param:
            param['mu'] = 0

    # half-filling Heisenberg model
    elif model == 'heis':
        if 't' not in param:
            param['t'] = 1
        if 'beta' not in param:
            param['beta'] = 1
        if 'mu' not in param:
            param['mu'] = 0

    # common parameters
    if 'eps' not in param and param['shrink1by1'] is False:
        param['eps'] = param['beta'] / 2**param['verticalStep']
    elif 'eps' in param and param['shrink1by1'] is True:
        param['verticalStep'] = int(round(param['beta'] / param['eps']))
        param['beta'] = param['verticalStep'] * param['eps']

    # consistency check
    assert not (param['h3to1'] and param['h2to1'])
    if param['3body']:
        assert param['mpo'] and param['h3to1'] and not param['h2to1'] and not param['3bodyAB']
        param['eps'] *= 1.5
    else:
        if param['3bodyAB']: assert param['mpo']
        if param['mpo']: param['eps'] /= 2.0
        if param['h3to1']: param['eps'] *= 3.0
        if param['h2to1']: param['eps'] *= 2.0
    return param
