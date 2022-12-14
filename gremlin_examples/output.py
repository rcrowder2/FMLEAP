#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 09:32:29 2022

@author: robert
"""
from leap_ec.global_vars import context
from leap_ec import probe

import os
import pandas as pd


def p3_output(config, context, parents):
    
    bsf = probe.best_of_gen(parents)
    print(f'fitness: {bsf.fitness}')
    print(f'keff: {bsf.k}')
    
    outfile = config.output.filename
    
    if not os.path.isfile(outfile):
        N_list = []
        E_list = []
        mut_list = []
        xo_list = []
        test_list = []
        gen_list = []
        best_list = []
        k_list =  []
        msek_list = []
        genome_list = []
        
        data = {'population': N_list,
            'elites': E_list,
            'mutation prob': mut_list,
            'xo prob': xo_list,
            'gen': gen_list,
            'best': best_list,
            'k': k_list,
            # 'mse k': msek_list,
            'genome': genome_list}
        
        df = pd.DataFrame.from_dict(data)
        
        df.to_pickle(outfile)
        
    
    
    df = pd.read_pickle(outfile)
    
    df.loc[len(df.index)] = [config.standard_params.pop_size,
                             config.standard_params.k_elites,
                             config.standard_params.mutation.p_mut,
                             config.standard_params.crossover.p_c,
                             # config.run_id,
                             context['leap']['generation'],
                             probe.best_of_gen(parents).fitness,
                             probe.best_of_gen(parents).k,
                             # mse_k,
                             probe.best_of_gen(parents).genome]
    
    df.to_pickle(outfile)