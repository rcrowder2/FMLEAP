#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:20:02 2022

@author: robert

Functions used to create the XML file for openmc runs
"""

import openmc as om
import warnings
import math
import numpy as np
import random


def build_xmls_p2(genome):
    """
    Creates the necessary xml files for openmc for a given genome. Edit the function
    for whichever geometry is needed

    Parameters
    ----------
    genome : array_like
        array of genes to be applied to openmc geometry.

    Returns
    -------
    None.

    """

    # warnings.filterwarnings("ignore")
    
    assert len(genome) == 11**2
    num_cells = 11
    
    problem_length = 55

######################## Create materials.xml #################################
    mats = []

    for i,rho in enumerate(genome):
        
        if rho > 0:
            mat = om.Material()#material_id = i+1)
            mat.add_element('U', 1, enrichment=3)
            mat.add_element('O', 2)
            mat.set_density('g/cm3', density=10.4*rho)
            
            mod = om.Material()
            mod.add_element('H', 2)
            mod.add_element('O', 1)
            mod.set_density('g/cm3', density=1*rho)
            
            mix = om.Material.mix_materials([mat,mod], [0.227, 0.773], percent_type='ao')
            mix.id = i
            
        else:
            mix = None
        
        mats.append(mix)
            
    materials = om.Materials(materials=[mat for mat in mats if mat is not None])

    materials.export_to_xml()

######################## Create geometry.xml ##################################
    xs = np.linspace(0,problem_length,num_cells+1)
    xSurfs = [om.XPlane(x0=x) for x in xs]
    ySurfs = [om.YPlane(y0=x) for x in xs]
    xSurfs[0].boundary_type='vacuum'
    xSurfs[-1].boundary_type='vacuum'
    ySurfs[0].boundary_type='vacuum'
    ySurfs[-1].boundary_type='vacuum'

    Cells = []

    for idy in range(num_cells):
        for idx in range(num_cells):
            Cells.append(
                om.Cell(fill = mats[idy*num_cells + idx],
                        region = +xSurfs[idx] & -xSurfs[idx+1] & +ySurfs[idy]
                                & -ySurfs[idy+1],
                        cell_id = idy*num_cells + idx)
                )

    geometry = om.Geometry(Cells)
    geometry.export_to_xml()

######################## Create settings.xml ##################################
    settings = om.Settings()
    batches = 100
    settings.batches = batches
    settings.inactive = 10
    settings.particles = 1000
    settings.seed = random.randrange(0,1e9)
    settings.output = {'tallies': False}

    bounds = [0, 0, 0, problem_length, problem_length, problem_length]
    uniform_dist = om.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
    settings.source = om.source.Source(space=uniform_dist)

    settings.verbosity = 1

    settings.export_to_xml()

    warnings.resetwarnings()


def build_tallies_p2():

    geometry = om.Geometry.from_xml()
    # geometry = geometry.from_xml()

    Cells = geometry.get_all_cells()

    Cells = list(Cells.values())

    cell_filter = om.CellFilter(Cells, filter_id=1)

    cellborn_filter = om.CellbornFilter(Cells, filter_id=2)

    tally = om.Tally(tally_id=1)
    tally.filters = [cell_filter, cellborn_filter]
    tally.scores = ['nu-fission']
    tally.estimator = 'collision'


    tallies_file = om.Tallies([tally])
    tallies_file.export_to_xml()

def create_source(genome):

    #TODO: Create more intelligent source based off genome to avoid sampling errors
    pass

if __name__ == '__main__':

    genome = np.random.random((11**2))

    build_xmls_p2(genome)
    build_tallies_p2()

    om.run()

    import Read_results as rr

    k,*res = rr.read_FM_results()