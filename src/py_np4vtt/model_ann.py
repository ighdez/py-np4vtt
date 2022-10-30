#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Modules to configure and estimate an ANN-based VTT model."""
from dataclasses import dataclass

from typing import List, Optional

from numpy import ndarray

from py_np4vtt.data_format import ModelArrays

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# TODO: Check with Joao how safe is to import os

@dataclass
class ConfigANN:
    """Configuration class of the ANN-based model.
    
    This class stores the configuration parameters of an ANN-based model 
    and performs integrity checks before being passed to the model object.
    
    Parameters
    ----------
    hiddenLayerNodes : List[int]
        Topology of the ANN. The length of `hiddenLayerNodes` is the number 
        of hidden layers, and each element of the list is the number of 
        nodes of the corresponding layer.
    trainingRepeats : int
        Number of estimation repeats of the ANN.
    shufflesPerRepeat : int
        Random shuffles of the data per repetition.

    seed : Optional[int]
        Random seed for the shuffling and estimation process.
    """
    hiddenLayerNodes: List[int]

    trainingRepeats: int
    shufflesPerRepeat: int

    seed: Optional[int]

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.trainingRepeats > 0:
            errorList.append('Number of repeats must be positive.')

        if not self.shufflesPerRepeat > 0:
            errorList.append('Number of shuffles per repeats must be positive.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelANN:
    """ANN model.
    
    This is the model class that prepares the data and estimates 
    the ANN-based model [1].
    
    Parameters
    -----------
    params : ConfigANN
        A configuration class of an ANN-based model.
    arrays : ModelArrays
        Model arrays created with `make_modelarrays`
        
    Attributes
    ----------
    X_train : numpy.ndarray
        Input data for training (estimation), i.e. the BVTT of each choice set and the T-1'th choices
    X_test : numpy.ndarray
        Input data for testing (prediction)
    X_full : numpy.ndarray
        Full input data
    y_train : numpy.ndarray
        Output data for training, i.e. the T'th choice
    y_test : numpy.ndarray
        Output data for testing
    y_full : numpy.ndarray
        Full output data

    Methods
    -------
    run():
        Estimates the ANN-based model.
    
    References
    ----------
    [1] van Cranenburgh, Sander, and Marco Kouwenhoven. "An artificial neural 
    network based method to uncover the value-of-travel-time distribution." 
    Transportation 48.5 (2021): 2545-2583.
    """
    def __init__(self, cfg: ConfigANN, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

        # Initialise arrays for randomisation
        shuffle_index = np.zeros((self.cfg.shufflesPerRepeat,self.arrays.T+1))
        full_data_array = np.zeros((self.arrays.NP,self.cfg.shufflesPerRepeat*(self.arrays.T+1),2))

        # Randomise data
        for n in range(self.arrays.NP):
            for m in range(self.cfg.shufflesPerRepeat):
                rnd11 = np.arange(self.arrays.T)
                np.random.shuffle(rnd11)
                shuffle_index[m,:] = np.hstack((rnd11,rnd11[np.random.randint(1,self.arrays.T-1)]))
            
            full_data_array[n,:,0] = self.arrays.Choice[n,shuffle_index.astype(int).flatten()]
            full_data_array[n,:,1] = self.arrays.BVTT[n,shuffle_index.astype(int).flatten()]

        # Create input and output arrays for ANN
        full_data_array = np.hstack((np.reshape(full_data_array[:,:,0].T,(self.arrays.T+1,self.cfg.shufflesPerRepeat*self.arrays.NP),order='F').T,np.reshape(full_data_array[:,:,1].T,(self.arrays.T+1,self.cfg.shufflesPerRepeat*self.arrays.NP),order='F').T))
        t = full_data_array[:,0]
        x = full_data_array[:,1:]

        # Separate in train and test
        X_train, X_test, y_train, y_test = train_test_split(x,t,test_size = 0.15,random_state=self.cfg.seed)

        self.X_train = X_train
        self.X_test = X_test
        self.X_full = x
        self.y_train = y_train
        self.y_test = y_test
        self.y_full = t

    def run(self, verbose=True):
        """Estimates the Rouwendal model.
        
        Parameters
        ----------
        None.

        Returns
        -------
        ll_list : np.ndarray
            The log-likelihood values per repetition.
        r2_list : np.ndarray
            The Rho-squared values per repetition.
        vtt_list : np.ndarray
            The estimated VTT per respondent and repetition.
        """
        ll_list = []
        rho_sq = []
        y_predict = []
        VTT_mid_list = []

        for r in range(self.cfg.trainingRepeats):
            
            if verbose:
                print('Rep ' + str(r+1) + ': ',end='',flush=True)

            clf = MLPClassifier(
                hidden_layer_sizes=self.cfg.hiddenLayerNodes,
                activation='tanh',
                tol=1e-4,
                alpha=0.,
                n_iter_no_change=6,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1275,
                verbose=False,random_state=self.cfg.seed).fit(self.X_train,self.y_train)

            # Predict in test sample
            y_predict_train = clf.predict_proba(self.X_train)
            y_predict_test = clf.predict_proba(self.X_test)

            y_predict.append(y_predict_test)

            # Get train and test loss
            train_loss = log_loss(self.y_train,y_predict_train)
            test_loss = log_loss(self.y_test,y_predict_test)

            # Compute log-likelihood and Rho-sq in full sample
            ll = -len(self.y_train)*train_loss - len(self.y_test)*test_loss
            r2 = 1 - (ll/(np.log(0.5)*len(self.y_full)))
            ll_list.append(ll)
            rho_sq.append(r2)

            if verbose:
                print('CE (train): ' + str(round(train_loss,4)) + ' / CE (test): ' + str(round(test_loss,4)) + ' / LL: ' + str(round(ll,2)) + ' / Rho-sq: ' + str(round(r2,2)))
            
            # Simulate N-choice of each individual using the ANN
            vtt_grid = np.tile(np.linspace(0,1.5*self.X_full.max(),101),(self.arrays.Accepts.shape[0],1))
            y_pred_N = ModelANN.simulateNChoice(self,clf,self.arrays.Choice,vtt_grid,self.arrays.BVTT,20)

            # Recover individual VTTs, using simulation of choice probs
            VTT_mid = np.zeros(self.arrays.NP)
            for n in range(self.arrays.NP):
                if np.max(y_pred_N[n,:]) > 0.5 and np.min(y_pred_N[n,:])<0.5:
                    delta_x = vtt_grid[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - vtt_grid[n,np.where((y_pred_N[n,:]-0.5)<=0)[0][0]]
                    delta_y = y_pred_N[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - y_pred_N[n,np.where((y_pred_N[n,:]-0.5)<=0)[0][0]]
                    VTT_mid[n] = vtt_grid[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - (y_pred_N[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]]-0.5)/(delta_y/delta_x)

            VTT_mid_list.append(VTT_mid)

        ll_list = np.array(ll_list)
        r2_list = np.array(rho_sq)
        vtt_list = np.array(VTT_mid_list)

        return ll_list, r2_list, vtt_list

    @staticmethod
    def simulateNChoice(self,clf,y,vtt_grid,X,R):

        # Create arrays for random shufflings
        x_sim = np.zeros((y.shape[0]*vtt_grid.shape[1],2*self.arrays.T+1,R))
        y_sim = np.zeros((y.shape[0],vtt_grid.shape[1],R))
        
        for n in range(y.shape[0]):
            for r in range(R):
                rndp = np.arange(self.arrays.T)
                np.random.shuffle(rndp)

                x_sim[(n*vtt_grid.shape[1]):((n+1)*vtt_grid.shape[1]),:,r] = np.c_[np.tile(y[n,rndp],(vtt_grid.shape[1],1)), vtt_grid[n,:].T, np.tile(X[n,rndp],(vtt_grid.shape[1],1))]
        
        for r in range(R):
            y_sim[:,:,r] = np.reshape(clf.predict_proba(x_sim[:,:,r])[:,1],(vtt_grid.shape[1],y.shape[0]),order='F').T
        y_median = np.median(y_sim,axis=2)

        return y_median