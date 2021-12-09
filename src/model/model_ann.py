#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import List, Optional

from model.data_format import ModelArrays

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

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
    y_train: np.ndarray
    y_test: np.ndarray
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
                shuffle_index[m,:] = np.hstack((rnd11,rnd11[np.random.randint(1,self.cfg.T-1)]))
            
            full_data_array[n,:,0] = self.arrays.Choice[n,shuffle_index.astype(int).flatten()]
            full_data_array[n,:,1] = self.arrays.BVTT[n,shuffle_index.astype(int).flatten()]

        # Create input and output arrays for ANN
        full_data_array = np.hstack((np.reshape(full_data_array[:,:,0].T,(self.arrays.T+1,self.cfg.shufflesPerRepeat*self.arrays.NP),order='F').T,np.reshape(xx[:,:,1].T,(K+1,R*N),order='F').T))
        t = full_data_array[:,0]
        x = full_data_array[:,1:]

        # Separate in train and test
        X_train, X_test, y_train, y_test = train_test_split(x,t,test_size = 0.15)

        initialArgs = InitialArgsANN(
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            ann_topology = tuple(self.cfg.hiddenLayerNodes))

        return initialArgs

    def run(self, args: InitialArgsANN) -> None:
        
        # for r in range(self.cfg.trainingRepeats):
        
        # # Setup and fit the ANN
        # clf = MLPClassifier(
        #     hidden_layer_sizes=args.ann_topology,
        #     activation='tanh',
        #     tol=1e-4,
        #     n_iter_no_change=6).fit(args.X_train,args.y_train)

        # # Predict in test sample
        # y_predict = clf.predict_proba(args.X_test)

        # # Get train and test loss
        # train_loss = clf.best_loss_
        # test_loss = log_loss(args.y_test,y_predict)

        # # Compute log-likelihood and Rho-sq in full sample
        # LL = -len(args.y_train)*train_loss - len(args.y_test)*test_loss
        # rho_sq = 1 - (LL/(np.log(0.5)*len(args.y_train)))

        pass