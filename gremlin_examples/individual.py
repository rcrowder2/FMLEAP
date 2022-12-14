#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:15:06 2022

@author: robert
"""
from leap_ec.individual import LibraryIndividual
from math import nan

class P2Individual(LibraryIndividual):
    
    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome,decoder,problem)
        self.mass = None
     
    def evaluate(self):

        """ Determine Individual's fitness and associated k_eff, fission distribution
        and fission matrix

        :return: calculated fitness, k_eff, fission distribution and fission matrix
        """

        try:
            self.fitness, self.k, self.FM, self.mass = self.evaluate_imp()
            self.is_viable = True  # we were able to evaluate
        except Exception as e:
            self.fitness = nan
            self.exception = e
            self.is_viable = False  # we could not complete an eval
            
        return self.fitness, self.k, self.FM, self.mass