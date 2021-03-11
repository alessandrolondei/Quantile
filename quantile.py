#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:56:41 2021

@author: alessandro
"""

import numpy as np

class Quantile(object):
    
    """
    A class to create and handle a quantile representation.
    A quantile can be created either by data or by simple quantile features
    (as number of bins or extension)

    ...

    Attributes
    ----------
    None

    Methods
    -------
    set_by_data(data, nbins, mode='constant', center='half', minvals=1):
        Creates a quantile representation by a list or numpy array.
        
        data: list or numpy array
        
        nbins: the number of bins
        
        mode: ['constant', 'frequence'] sets the bins either equally
              distributed ('constant') or by keeping the same amount of data
              into each bin ('frequence'). Default: 'constant'
              
        center: ['half', 'mean'] sets the bin center either in the half
              ('half') or on the mean of data containedin that bin ('mean').
              Default: 'half'
                
        minvals: applies only in case of center='mean'. If the number of data 
              points contained in the bin is less than minvals, the relative
              center is given by 'half'. Default: 1
                 
                 
    set_by_bins(bin_ticks, bin_centers=None):
        Creates a quantile representation by an array of bin ticks (the number
        of bins is bin_ticks - 1) and, eventually, by an array describing the
        bin centers.
        
        bin_ticks: an array containing the bin ticks. Bin ticks are the values 
               defining the inital point of every bin and the last closing
               point. The number of ticks is given by bin_ticks - 1.
                   
        bin_centers (optional): an array containing the bin centers. Default:
               None
               
    set_by_minmax(minmax, nbins):
        Creates a quantile representation by setting the min and max value, and
        the number of bins.
        
        minmax: is a tuple containing the min and the max value of the quantile
              representation.
              
        nbins: is the total number of bins.
        
        
    getbin(x): given a value or an array x, getbin returns the related  bin or 
        bins value. The bin value is an integer from 0 to bin max.
        
    getcenter(binvalue): given a bin value or an array binvalue, getcenter
        returns the related center(s). It is affected by the center feature
        ['half', 'mean']
    
    
    Usage
    -----
    
    q = Quantile()
    data = np.random.randn(2, 1000)
    q.set_by_data(data, 40, mode='constant', center='half', minvals=10)
    """
    
    def __init__(self):
        self.nbins = None
        self.data = None
        self.bin_ticks = None
        self.bin_centers = None
        self.mode = None
        self.center = None
        self.source = None
        self.minmax = None
        self.minvals = None
    
    
    def set_by_data(self, data, nbins, mode='constant', center='half', minvals=1):
        '''
        Creates a quantile representation by a list or numpy array.

        Parameters:
            data: list or numpy array
        
            nbins: the number of bins
            
            mode: ['constant', 'frequence'] sets the bins either equally
                  distributed ('constant') or by keeping the same amount of data
                  into each bin ('frequence'). Default: 'constant'
                  
            center: ['half', 'mean'] sets the bin center either in the half
                  ('half') or on the mean of data containedin that bin ('mean').
                  Default: 'half'
                    
            minvals: applies only in case of center='mean'. If the number of data 
                  points contained in the bin is less than minvals, the relative
                  center is given by 'half'. Default: 1
        '''
        self.source = 'data'
        self.nbins = nbins
        self.mode = mode
        self.minvals = minvals
        if self.mode not in ['constant', 'frequence']:
            raise ValueError('mode is wrong')
        self.center = center
        if self.center not in ['half', 'mean']:
            raise ValueError('center is wrong')
        
        self.data = np.array(data)
        
        datat = self.data.copy().flatten()
        datat = np.sort(datat)
        self.bin_ticks = np.zeros(self.nbins + 1)
        self.bin_centers = np.zeros(self.nbins)
        
        if mode == 'frequence':
            for k in np.arange(self.nbins):
                fr = int(k * len(datat) / self.nbins)
                to = int((k + 1) * len(datat) / self.nbins)
                self.bin_ticks[k] = datat[fr]
                if k == self.nbins - 1:
                    self.bin_ticks[k+1] = datat[-1]
                if center == 'half':
                    self.bin_centers[k] = (self.bin_ticks[k] + self.bin_ticks[k]+1) / 2
                elif center == 'mean':
                    self.bin_centers[k] = datat[fr:to].mean()
        elif mode == 'constant':
            self.bin_ticks = np.linspace(datat[0], datat[-1], num=nbins+1)
            if center == 'half':
                self.bin_centers = (self.bin_ticks[1:] + self.bin_ticks[:-1]) / 2
            elif center == 'mean':
                for k in np.arange(self.nbins):
                    ind = np.where(np.logical_and(datat>=self.bin_ticks[k], datat<=self.bin_ticks[k+1]))[0]
                    if len(ind) < minvals:
                        self.bin_centers[k] = (self.bin_ticks[k] + self.bin_ticks[k+1]) / 2
                    else:
                        self.bin_centers[k] = datat[ind].mean()
        self.minmax = (self.bin_ticks[0], self.bin_ticks[-1])
            
        
    
    def set_by_bins(self, bin_ticks, bin_centers=None):
        '''
        Creates a quantile representation by an array of bin ticks (the number
        of bins is bin_ticks - 1) and, eventually, by an array describing the
        bin centers.

        Parameters:
            bin_ticks: an array containing the bin ticks. Bin ticks are the values 
               defining the inital point of every bin and the last closing
               point. The number of ticks is given by bin_ticks - 1.
                   
        bin_centers (optional): an array containing the bin centers. Default:
               None

    '''
        self.source = 'bins'
        self.nbins = len(bin_ticks) - 1
        self.bin_ticks = np.array(bin_ticks)
        if bin_centers is None:
            self.bin_centers = (self.bin_ticks[1:] + self.bin_ticks[:-1]) / 2
        else:
            bin_centers = np.array(bin_centers)
            if len(bin_centers) != len(bin_ticks) - 1:
                raise ValueError('bin_centers len must be equal to bin_ticks len minus 1')
            found = False
            for k, bc in enumerate(bin_centers):
                if (bc < self.bin_ticks[k]) or (bc > self.bin_ticks[k+1]):
                    found = True
            if found:
                raise ValueError('Each element in bin_centers must be inside the relative interval given by bin_ticks')
            self.bin_centers = bin_centers
        self.minmax = (self.bin_ticks[0], self.bin_ticks[-1])
    
    def set_by_minmax(self, minmax, nbins):
        '''
        Creates a quantile representation by setting the min and max value, and
        the number of bins.

        Parameters:
            minmax: is a tuple containing the min and the max value of the quantile
              representation.
              
            nbins: is the total number of bins.

        '''
        self.source = 'minmax'
        self.nbins = nbins
        self.bin_ticks = np.linspace(minmax[0], minmax[1], num=nbins+1)
        self.bin_centers = (self.bin_ticks[1:] + self.bin_ticks[:-1]) / 2
        self.minmax = (self.bin_ticks[0], self.bin_ticks[-1])
    
    
    def getbin(self, x):
        '''
        given a value or an array x, getbin returns the related  bin or 
        bins value. The bin value is an integer from 0 to bin max.

        Parameters:
            x (float or array): A data point or array

        Returns:
            binvalue (int or array): The related bin value(s)
    '''
        if isinstance(x, (int, float)):
            binvalue = (x > self.bin_ticks[:-1]).sum()
            if binvalue == 0:
                binvalue = 1
            return binvalue - 1
        else:
            x = np.array(x)
            binvalue = np.zeros_like(x).astype(int)
            for k, y in enumerate(x):
                binvalue[k] = (y > self.bin_ticks[:-1]).sum()
                if binvalue[k] == 0:
                    binvalue[k] = 1
                binvalue[k] -= 1
            return binvalue
        
    def getcenter(self, binvalue):
        '''
        given a bin value or an array binvalue, getcenter
        returns the related center(s). It is affected by the center feature
        ['half', 'mean']

        Parameters:
            binvalue (int or array): The bin value(s)

        Returns:
            centers (float or array): The related center(s)
        '''
        if isinstance(binvalue, (int, float)):
            return self.center[binvalue]
        else:
            binvalue = np.array(binvalue)
            centers = np.zeros_like(binvalue).astype(int)
            for k, y in enumerate(binvalue):
                centers[k] = self.center[y]
            return centers


        
        
        
