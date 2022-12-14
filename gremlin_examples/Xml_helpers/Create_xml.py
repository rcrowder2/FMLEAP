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



max_packing_fraction = 0.74

# nominal graphite moderator number density based off of max theoretical density
density_graphite = 2.23
N_graphite = density_graphite/12.01*6.022e23/1e24


# nominal homogenized triso as max packing fraction. 
N_triso = 0.08370595415166099*max_packing_fraction

N_uranium = 0.003839001053998391*max_packing_fraction
N_silcon = 0.008846355031990744*max_packing_fraction
N_carbon_triso = 0.07102059806567185*max_packing_fraction


def build_xmls(genome, problem_length):
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

    warnings.filterwarnings("ignore")
    packing_fractions = genome[0]
    rhos_graphite = genome[1]

    num_cells = int(math.sqrt(len(packing_fractions)))

######################## Create materials.xml #################################
    mats = []
    fissionable = []

    for pf, rho in zip (packing_fractions,rhos_graphite):

        if pf != 0 and rho != 0:

            N_total = pf*N_triso + rho*N_graphite

            mat = om.Material()
            mat.set_density('atom/b-cm', N_total)
            mat.add_element('U',pf*N_uranium, enrichment=19.95)
            mat.add_element('Si', pf*N_silcon)
            mat.add_element('C', pf*N_carbon_triso + rho*N_graphite)

            mat.add_s_alpha_beta('c_Graphite')

            mat.id = len(mats)
            mats.append(mat)
            
            fissionable.append(True)

        elif pf != 0:

            mat = om.Material()
            mat.set_density('atom/b-cm',pf*N_triso)
            mat.add_element('U',pf*N_uranium, enrichment=19.95)
            mat.add_element('Si', pf*N_silcon)
            mat.add_element('C', pf*N_carbon_triso)

            mat.add_s_alpha_beta('c_Graphite')

            mat.id = len(mats)
            mats.append(mat)
            
            fissionable.append(True)

        elif rho != 0:

            graphite = om.Material()
            graphite.set_density('atom/b-cm',rho*N_graphite)
            graphite.add_element('C', 1)

            graphite.add_s_alpha_beta('c_Graphite')

            graphite.id = len(mats)
            mats.append(graphite)
            
            fissionable.append(False)

        else:
            mats.append(None)
            fissionable.append(False)


    materials = om.Materials(materials=[mat for mat in mats if mat is not None])

    materials.export_to_xml()

######################## Create geometry.xml ##################################
    xs = np.linspace(0,problem_length,num_cells+1)
    xSurfs = [om.XPlane(x0=x) for x in xs]
    ySurfs = [om.YPlane(y0=x) for x in xs]
    xSurfs[0].boundary_type='reflective'
    xSurfs[-1].boundary_type='reflective'
    ySurfs[0].boundary_type='reflective'
    ySurfs[-1].boundary_type='reflective'

    Cells = []

    for idy in range(num_cells):
        for idx in range(num_cells):
            
            cell_id = idy*num_cells + idx
            Cells.append(
                om.Cell(fill = mats[cell_id],
                        region = +xSurfs[idx] & -xSurfs[idx+1] & +ySurfs[idy]
                                & -ySurfs[idy+1],
                        cell_id = cell_id)
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

    uniform_dist = create_source_dist(genome[0], num_cells, problem_length)
    settings.source = om.source.Source(space=uniform_dist)

    settings.verbosity = 1

    settings.export_to_xml()

    warnings.resetwarnings()


def build_tallies():

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

def create_source_dist(genome, num_cells, problem_length):

    #TODO: Create more intelligent source based off genome to avoid sampling errors
    xlow,ylow = num_cells,num_cells
    xhigh,yhigh = 0,0
    
    for y in range(num_cells):
        for x in range(num_cells):
            cell = y*num_cells + x
            
            if genome[cell] > 0:
                
                if x < xlow: xlow = x
                if y < ylow: ylow = y
                if x > xhigh: xhigh = x
                if y > yhigh: yhigh = y
                
    surfs = np.linspace(0,problem_length,num_cells+1)
    bounds = [surfs[xlow], surfs[ylow], 0, surfs[xhigh+1], surfs[yhigh + 1], 1]
    
    return om.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
            
            

if __name__ == '__main__':

    genome = np.array([0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0]).reshape((2,16))

    build_xmls(genome,1)
    build_tallies()

    om.run()

    import Read_results as rr

    print(rr.read_FM_results())