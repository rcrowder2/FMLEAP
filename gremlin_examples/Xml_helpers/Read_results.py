#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:13:30 2022

@author: robert

Functions used to parse openmc evaluations
"""

import openmc as om
import glob, os
import numpy as np
from numpy.linalg import eig


def read_std_results():

    batches = om.Settings.from_xml().batches

    sp = om.StatePoint(f'statepoint.{batches}.h5')

    k = sp.k_combined.nominal_value

    sp._f.close()

    remove_output_files()

    return k

def read_FM_results():

    batches = om.Settings.from_xml().batches

    sp = om.StatePoint(f'statepoint.{batches}.h5')

    k = sp.k_combined.nominal_value

    t = sp.get_tally(id=1)
    nu_fission = t.mean
    num_cells = int(np.sqrt(len(nu_fission)))
    nu_fission_matrix = nu_fission.reshape((num_cells,num_cells))

    b = np.diag(k/sum(nu_fission_matrix.transpose()))
    b[np.isnan(b) | np.isinf(b)] = 0
    FM = np.matmul(nu_fission_matrix,b)

    _,fdist = eig(FM)

    fdist = fdist[:,0]/np.sum(fdist[:,0])

    sp._f.close()

    remove_output_files()

    return k, fdist.real, FM

def remove_output_files():

    for f in glob.glob("*.h5"):
        os.remove(f)
    for f in glob.glob("*.xml"):
        os.remove(f)