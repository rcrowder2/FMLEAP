#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 08:53:40 2022

@author: robert
"""


from leap_ec.problem import ScalarProblem
from leap_ec import context

from Xml_helpers.Create_xml import build_tallies,build_xmls
from Xml_helpers.Create_xml_p2 import build_tallies_p2,build_xmls_p2
import Xml_helpers.Read_results as Read_results

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import openmc as om

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import eig, norm
import numpy.matlib
from math import exp
import warnings


class FmgaProblem3(ScalarProblem):
    
    problem_length = None
    def __init__(self, problem_length):
        
        FmgaProblem3.problem_length = problem_length
        
        super().__init__(maximize=True)

    def pre_evaluate(self, ind):

        library = context['leap']['Library']

        genomes = [indv.decode() for indv in library]
        genomes = [indv.flatten() for indv in genomes]
        genomes = np.array(genomes)

        dist = norm(genomes - ind.flatten(),axis=1)

        idx = np.argpartition(dist,2)

        p1,p2 = library[idx[0]], library[idx[1]]

        k = np.real(FmgaProblem3.FMinterpolation(ind, p1, p2))

        fitness = k

        return fitness, k

    def evaluate(self, ind):
        
        build_xmls(ind, self.problem_length)
        build_tallies()

        om.run()

        k, fiss_dist, FM = Read_results.read_FM_results()

        fitness = k
        return fitness, k, FM

    @staticmethod
    def FMinterpolation(x,p1,p2):

        p1_genome = p1.decode()
        p2_genome = p2.decode()

        I = len(x[0])
        pfs = np.zeros((I,I,3))
        rhos = np.zeros((I,I,3))

        pfs[:,:,0] = np.matlib.repmat(p1_genome[0],I,1).transpose()
        pfs[:,:,1] = x[0]
        pfs[:,:,2] = np.matlib.repmat(p2_genome[0],I,1).transpose()
        pfs[:,:,1] = pfs[:,:,1].transpose()


        rhos[:,:,0] = np.matlib.repmat(p1_genome[1],I,1).transpose()
        rhos[:,:,1] = x[1]
        rhos[:,:,2] = np.matlib.repmat(p2_genome[1],I,1).transpose()
        rhos[:,:,1] = rhos[:,:,1].transpose()


        # calculate alphas
        alphasf = (pfs[:,:,1] - pfs[:,:,0])/(pfs[:,:,2] - pfs[:,:,0])
        alphasf[np.isnan(alphasf) | np.isinf(alphasf)] = 0
        # calculate FMf
        FMf = (1-alphasf)*p1.FM + alphasf*p2.FM
        # FMf[FMf<0] = 0

        alphasc = (rhos[:,:,1] - rhos[:,:,0])/(rhos[:,:,2] - rhos[:,:,0])
        alphasc[np.isnan(alphasc) | np.isinf(alphasc)] = 0
        # calculate FMf
        FMc = (1-alphasc)*p1.FM  + alphasc*p2.FM
        # FMc[FMc<0] = 0

        f_dist = abs(pfs[:,:,2] - pfs[:,:,0])
        c_dist = abs(rhos[:,:,2] - rhos[:,:,0])
        dist_sum = f_dist + c_dist

        FM = (f_dist/dist_sum)*FMf + (c_dist/dist_sum)*FMc
        FM[FM<0] = 0

        for idx,row in enumerate(FM):
            if np.isnan(row).any():
                FM[idx] = FMf[idx]
        # eigen
        ks,dists = eig(FM)

        k = np.real(ks[0])


        return k



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 3).double() 
        self.l1.weight.data.fill_(1)
        self.l2 = nn.Linear(3, 3).double()
        self.l2.weight.data.fill_(1)
        self.l3 = nn.Linear(3, 1).double()
        self.l3.weight.data.fill_(1)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
    
    
class FmgaProblem2(ScalarProblem):
    
    model = Model()
    model.load_state_dict(torch.load('1D_NN_v3.pth'))
    model.eval()
    
    epoch = 0
    
    def __init__(self):
        
        super().__init__(maximize=False)
        
        

    def evaluate(self, ind):
        
        mass = sum(ind)
       
        build_xmls_p2(ind)
        build_tallies_p2()
        
        warnings.filterwarnings("ignore")
        om.run()
        warnings.resetwarnings()

        k, fdist, FM = Read_results.read_FM_results()

        fitness = mass
        
        if k < 1:
            fitness += exp(100*(1-k))
            
        return fitness, k, FM, mass
        
    def pre_evaluate(self, ind):
        
        library = context['leap']['Library']

        genomes = [indv.decode() for indv in library]
        
        genomes = np.array(genomes)

        dist = norm(genomes - ind,axis=1)

        idx = np.argpartition(dist,2)

        p1,p2 = library[idx[0]], library[idx[1]]

        fitness, k = np.real(FmgaProblem2.FMinterpolation(ind, p1, p2))

        return fitness, k
    
    @classmethod
    def train_model(cls):
        
        children = context['leap']['new']
        parents_list = cls.find_closest_parents(children)
        
        dataset = GADataset(children, parents_list)
        
        dataset.plot_dataset()
        
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
        
        
        train_loader = DataLoader(dataset=train_dataset,
                      batch_size=len(train_dataset)//len(children),
                      shuffle=True,
                      num_workers=2)
        
        test_loader = DataLoader(dataset=train_dataset,
                      batch_size=len(test_dataset)//len(children),
                      shuffle=True,
                      num_workers=2)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(FmgaProblem2.model.parameters(), lr=1e-4)
        
        num_epochs = 25
        
        for epoch in range(cls.epoch,cls.epoch + num_epochs):
            epoch_loss = 0
            # acc = 0
            for i, (alphas, betas) in enumerate(train_loader):
                
                optimizer.zero_grad()
                
                betas_pred = cls.model(alphas)
                loss = criterion(betas_pred, betas)
                
                loss.backward()
                
                optimizer.step()
                
                epoch_loss += loss
                
            print(f'{epoch} | {epoch_loss =:.4f} ')
                
            cls.writer.add_scalar("loss", epoch_loss, epoch)
            
            if (epoch+1) % 10 == 0:
                
                
                
                for name, values in cls.model.named_parameters():
                    cls.writer.add_histogram(name, values,  epoch)
                
        cls.epoch += num_epochs
        
        
            
            #TODO: figure out the best way to validate the training.
            # with torch.no_grad():
            #     for j, (alphas, betas) in enumerate(train_loader):
                    
            #         betas_pred = FmgaProblem2.model(alphas)
            #         loss = criterion(betas_pred, betas)
                    
            #         acc += loss
                    
            #     acc /= (j+1)
                
                            
     
    @staticmethod
    def find_closest_parents(children):
        
        library = context['leap']['Library']
        parent_genomes = [indv.decode() for indv in library]
        child_genomes = [indv.decode() for indv in children]
        
        parents_list = []
        
        for child_genome in child_genomes:
            
            dist = norm(parent_genomes - child_genome,axis=1)

            idx = np.argpartition(dist,2)

            parents_list.append((library[idx[0]], library[idx[1]]))
            
        return parents_list
        
    @staticmethod
    def FMinterpolation(x,p1,p2):
        
        I = len(x)

        alphas_tensor = FmgaProblem2.calculate_alphas(x, p1, p2)        
        
        betas_tensor = FmgaProblem2.model(alphas_tensor)
        
        betas_np = betas_tensor.detach().numpy().reshape((I,I))
        
        FM1 = p1.FM
        FM2 = p2.FM
        
        FM = (FM2-FM1)*betas_np + FM1
        
        FM[FM<0] = 0

        ks,dists = eig(FM)
        
        k = np.real(ks[0])
        
        fitness = sum(x)
        
        if k < 1:
            fitness += exp(100*(1-k))
        
        return fitness, k
    
    @staticmethod
    def calculate_alphas(x,p1,p2):
        
        p1_genome = p1.decode()
        p2_genome = p2.decode()

        I = len(x)
        
        rhos = np.zeros((I,I,3))

        rhos[:,:,0] = np.matlib.repmat(p1_genome,I,1).transpose()
        rhos[:,:,1] = x
        rhos[:,:,2] = np.matlib.repmat(p2_genome,I,1).transpose()
        rhos[:,:,1] = rhos[:,:,1].transpose()
        
        alphas_np = (rhos[:,:,1] - rhos[:,:,0])/(rhos[:,:,2] - rhos[:,:,0])
        
        alphas_np[np.isnan(alphas_np) | np.isinf(alphas_np)] = 0
        
        alphas_tensor = torch.from_numpy(alphas_np.flatten())
        alphas_tensor.unsqueeze_(1)
        
        return alphas_tensor
    
    @staticmethod
    def calculate_betas(FMc,FM1,FM2):
        
        betas_np = (FMc - FM1)/(FM2 - FM1)
        betas_np[np.isnan(betas_np) | np.isinf(betas_np)] = 0
        
        betas_tensor = torch.from_numpy(betas_np.flatten())
        betas_tensor.unsqueeze_(1)
        
        return betas_tensor
    
class GADataset(Dataset):
    
    def __init__(self,children,parents_list):
        
        self.n_samples = len(children)*len(children[0].genome)**2
        
        self.x_data = None
        self.y_data = None
        
        for child, parents in zip(children,parents_list):
            if self.x_data is not None :
                self.x_data = torch.cat(
                    (
                     self.x_data,
                     FmgaProblem2.calculate_alphas(child.decode(), parents[0], parents[1])
                    )
                    )
                
                self.y_data = torch.cat(
                    (
                     self.y_data,
                     FmgaProblem2.calculate_betas(child.FM, parents[0].FM, parents[1].FM)
                     )
                    )
            else:
                self.x_data = FmgaProblem2.calculate_alphas(child.decode(), parents[0], parents[1])
                self.y_data = FmgaProblem2.calculate_betas(child.FM, parents[0].FM, parents[1].FM)
                
    def plot_dataset(self):
        
        alphas, betas = self[:]
        
        fig = plt.figure()
        torch.Tensor.ndim = property(lambda self: len(self.shape))
        plt.scatter(alphas, betas)
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.title(f"gen: {context['leap']['generation']}")
        plt.show()
        
                
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.n_samples