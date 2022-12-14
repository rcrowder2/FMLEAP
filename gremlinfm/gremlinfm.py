#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gremlinfm.py
usage: gremlin.py [-h] [-d] config_files [config_files ...]
Gremlin finds features sets where a given machine learning model performs
poorly.
positional arguments:
  config_files  path to configuration file(s) which Gremlin uses to set up the
                problem and algorithm
optional arguments:
  -h, --help    show this help message and exit
  -d, --debug   enable debugging output
"""
import sys

# So we can pick up local modules defined in the YAML config file.
sys.path.append('.')

import argparse

import importlib

from omegaconf import OmegaConf


from leap_ec.global_vars import context
from leap_ec import ops, util
from leap_ec.int_rep.ops import mutate_randint, mutate_binomial
from leap_ec.real_rep.ops import mutate_gaussian, genome_mutate_gaussian, mutate_uniform
from leap_ec.segmented_rep.ops import add_segment, remove_segment, apply_mutation


from toolz import pipe


def read_config_file(config_file):
    """  Read one or more YAML files containing configuration options.
    The notion is that you can have a set of YAML files for controlling the
    configuration, such as having a set of default global settings that are
    overridden or extended by subsequent configuration files.
    E.g.,
    gremlin.py general.yaml this_model.yaml
    :param config_files: command line arguments
    :return: config object of current config
    """
    
    default_config = OmegaConf.load('default.yaml')
    user_config = OmegaConf.load(config_file)
    config = OmegaConf.merge(default_config,user_config)

    return config


def parse_config(config):
    """ Extract the population size, maximum generations to run, the Problem
    subclass, and the Representation subclass from the given `config` object.
    :param config: OmegaConf configurations read from YAML files
    :returns: Problem objects, Representation objects, LEAP pipeline operators,
        and optional with_client_exec_str
    """

    if config.imports is not None:
        # This allows for imports and defining functions referred to later in
        # the pipeline
        exec(config.imports, globals())
        
    if config.output is not None:
        globals()['output'] = importlib.import_module(config.output.function.split('.')[0])
        output_func = eval(config.output.function)
    else:
        output_func = None

    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    globals()['problem'] = importlib.import_module(config.problem.split('.')[0])
    globals()['representation'] = importlib.import_module(
        config.representation.split('.')[0])

    # Now instantiate the problem and representation objects, including any
    # ctor arguments.
    problem_obj = eval(config.problem)
    representation_obj = eval(config.representation)

    # Eval each pipeline function to build the LEAP operator pipeline
    pipeline = [eval(x) for x in config.pipeline]

    return problem_obj, representation_obj, pipeline, output_func



def run_std_ga(pop_size, max_generations, problem, representation,
                        pipeline, k_elites, output_func):
    
    """ Run a standard GA 
    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used in EA
    :param pop_file: where to write the population CSV file
    :param k_elites: keep k elites
    :param output_func: function for controlling the output data and files
    :returns: None
    """
    
    parents = representation.create_population(pop_size, problem=problem)

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(context=context)

    # Evaluate initial population
    parents = representation.individual_cls.evaluate_population(parents)

    generation_counter()

    while (generation_counter.generation() < max_generations):
        # Execute the operators to create a new offspring population
        parents = pipe(parents, *pipeline)
        
        output_func(config, context, parents)
        
        generation_counter()
        
        
def run_lib_ga(pop_size, max_generations, problem, representation,
                        pipeline, k_elites, best_frac, output_func):
    
    """ Run a FMGA with library method and possible neural network training
    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used in EA
    :param pop_file: where to write the population CSV file
    :param k_elites: keep k elites
    :param best_frac: parameter for library pooling
    :param output_func: function for controlling the output data and files
    :returns: None
    """
    
    # Taken from leap_ec.algorithm.generational_ea and modified pipeline
    # slightly to allow for printing population *after* elites are included
    # in survival selection to get accurate snapshot of parents for next
    # generation.

    # If birth_id is an attribute, print that column, too.
   

    

    # Initialize a population of pop_size individuals of the same type as
    # individual_cls
    parents = representation.create_population(pop_size, problem=problem)

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(context=context)

    # Evaluate initial population
    parents = representation.individual_cls.evaluate_population(parents)
    
    # add the population to the Library for use in interpolations
    context['leap']['Library'] = parents

    # print out the parents and increment gen counter
    generation_counter()

    while (generation_counter.generation() < max_generations):
        # Execute the operators to create a new offspring population
        parents = pipe(parents, *pipeline,
                         ops.library_survival(parents=parents,
                                              best_frac=best_frac, k=k_elites)
                         )
        
        parents[k_elites:] = representation.individual_cls.evaluate_population(parents[k_elites:])
        
        #train neural network model
        context['leap']['new'] = parents[k_elites:]
        if config.train_nn_model: problem.__class__.train_model()
        
        # add the new set of individuals to the Library
        context['leap']['Library'].extend(parents[k_elites:])
        
        # call the output function
        output_func(config, context, parents)
        
        generation_counter()  # Increment to the next generation


    if config.train_nn_model: problem.__class__.writer.close()

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
        description=('Gremlin finds features sets where a given machine '
                     'learning model performs poorly.'))

    parser.add_argument('config_file', type=str,
                        help=('path to configuration file which Gremlin '
                              'uses to set up the problem and algorithm'))
    args = parser.parse_args()

    # set logger to debug if flag is set
    
    # combine default and user configuration files into one dictionary
    config = read_config_file(args.config_file)
    

    # Import the Problem and Representation classes specified in the
    # config file(s) as well as the LEAP pipeline of operators
    problem, representation, pipeline, output_func = parse_config(config)

    pop_size = int(config.standard_params.pop_size)
    max_generations = int(config.standard_params.max_generations)
    k_elites = int(config.standard_params.k_elites) 
    
    if config.algorithm == 'lib':
        best_frac = config.library_params.best_fraction
        run_lib_ga(pop_size, max_generations, problem, representation,
                        pipeline, k_elites, best_frac, output_func)
    else:
        pass
        #TODO: make run_std_ga function
    

    