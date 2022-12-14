#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:34:33 2022

@author: robert
"""

import glob
import argparse
import pandas as pd

def main(baseFolderName, numFolders):
    
    df = pd.DataFrame()
    
    for file in glob.iglob(f'{baseFolderName}*/*.pickle'):
        temp_df = pd.read_pickle(file)
        df.append(temp_df)
        
    df.to_pickle(f'{baseFolderName}.pickle')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Combine GA runs into one pickled dataframe")
    parser.add_argument("--basefolder", "-b", required=True, type=str, help="Base folder name of GA runs")
    
    baseFolderName = parser.basefolder
    
    main()