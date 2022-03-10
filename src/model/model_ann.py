#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass

from typing import List, Optional, Tuple

from numpy import ndarray

from model.data_format import ModelArrays

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# TODO: Check with Joao how safe is to import os

@dataclass
class ConfigANN:
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


@dataclass
class InitialArgsANN:
    X_train: np.ndarray
    X_test: np.ndarray
    X_full: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_full: np.ndarray
    ann_topology: tuple

class ModelANN:
    def __init__(self, cfg: ConfigANN, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def setupInitialArgs(self) -> InitialArgsANN:

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
        X_train, X_test, y_train, y_test = train_test_split(x,t,test_size = 0.15)

        initialArgs = InitialArgsANN(
            X_train = X_train,
            X_test = X_test,
            X_full = x,
            y_train = y_train,
            y_test = y_test,
            y_full = t,
            ann_topology = self.cfg.hiddenLayerNodes)

        return initialArgs

    def run(self, args: InitialArgsANN, verbose=False) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        ll_list = []
        rho_sq = []
        y_predict = []
        VTT_mid_list = []

        for r in range(self.cfg.trainingRepeats):
            
            if verbose:
                print('Rep ' + str(r+1) + ': ',end='',flush=True)

            clf = MLPClassifier(
                hidden_layer_sizes=args.ann_topology,
                activation='tanh',
                tol=1e-4,
                alpha=0.,
                n_iter_no_change=6,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1275,
                verbose=False).fit(args.X_train,args.y_train)

            # Predict in test sample
            y_predict_train = clf.predict_proba(args.X_train)
            y_predict_test = clf.predict_proba(args.X_test)

            y_predict.append(y_predict_test)

            # # Get train and test loss
            train_loss = log_loss(args.y_train,y_predict_train)
            test_loss = log_loss(args.y_test,y_predict_test)

            # # Compute log-likelihood and Rho-sq in full sample
            ll = -len(args.y_train)*train_loss - len(args.y_test)*test_loss
            r2 = 1 - (ll/(np.log(0.5)*len(args.y_full)))
            ll_list.append(ll)
            rho_sq.append(r2)

            if verbose:
                print('CE (train): ' + str(round(train_loss,4)) + ' / CE (test): ' + str(round(test_loss,4)) + ' / LL: ' + str(round(ll,2)) + ' / Rho-sq: ' + str(round(r2,2)))
            
            # Simulate N-choice of each individual using the ANN
            vtt_grid = np.tile(np.linspace(0,1.5*args.X_full.max(),101),(self.arrays.Accepts.shape[0],1))
            y_pred_N = ModelANN.simulateNChoice(self,clf,self.arrays.Choice,vtt_grid,self.arrays.BVTT,20)

            # Recover individual VTTs, using simulation of choice probs
            VTT_mid = np.zeros(self.arrays.NP)
            for n in range(self.arrays.NP):
                if np.max(y_pred_N[n,:]) > 0.5 and np.min(y_pred_N[n,:])<0.5:
                    delta_x = vtt_grid[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - vtt_grid[n,np.where((y_pred_N[n,:]-0.5)<=0)[0][0]]
                    delta_y = y_pred_N[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - y_pred_N[n,np.where((y_pred_N[n,:]-0.5)<=0)[0][0]]
                    VTT_mid[n] = vtt_grid[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]] - (y_pred_N[n,np.where((y_pred_N[n,:]-0.5)>=0)[0][-1]]-0.5)/(delta_y/delta_x)

            VTT_mid_list.append(VTT_mid)

        return np.array(ll_list), np.array(rho_sq), np.array(y_predict), np.array(VTT_mid_list)

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