#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:48:33 2022

@author: robert
"""

from leap_ec.representation import Representation
from leap_ec.decoder import Decoder, IdentityDecoder
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.individual import LibraryIndividual
from individual import P2Individual

import numpy as np


class FmgaDecoder(Decoder):

    def __init__(self):
        super().__init__()


    def decode(self,genome,*args,**kwargs):

        assert len(genome)%2 == 0
        half = int(len(genome)/2)


        packing_fraction = genome[:half]
        rho_graphite = genome[half:]

        return np.vstack((packing_fraction,rho_graphite))


class FmgaRepresentation(Representation):
    
    # This says we have one gene that's an integer in the range [0,9].
    npixels = 5
    genome_bounds = (0,1)

    def __init__(self):
        super().__init__(
            initialize=create_binary_sequence(2*FmgaRepresentation.npixels**2),
            decoder=FmgaDecoder(),
            individual_cls=LibraryIndividual)
        


class P2Representation(Representation):
    
    # This says we have one gene that's an integer in the range [0,9].
    npixels = 11
    genome_bounds = (0,1)

    def __init__(self):
        super().__init__(
            initialize=create_binary_sequence(P2Representation.npixels**2),
            decoder=IdentityDecoder(),
            individual_cls=P2Individual)
        
