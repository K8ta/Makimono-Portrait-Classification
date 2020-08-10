import numpy as np
from imblearn.over_sampling import RandomOverSampler

class oversampled_Kfold():
    """ K-fold Cross ValidationとOverSampling """
    def __init__(self, n_splits, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits*self.n_repeats

    def split(self, X, y, groups=None):
        splits = np.array_split(np.random.choice(len(X), len(X),
                                replace=False).tolist(), self.n_splits)
        train, test = [], []

        for repeat in range(self.n_repeats):
            for idx in range(len(splits)):
                trainingIdx = np.concatenate(np.delete(splits, idx, axis=0))
                Xidx_r, y_r = RandomOverSampler().fit_resample(trainingIdx.reshape((-1,1)), 
                                                               y[trainingIdx].reshape((-1,1)))
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))