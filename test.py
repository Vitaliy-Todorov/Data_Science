# %% codecell

import numpy as np

class AdalineGD(object) :

    def __init__(self, eta = .01, n_iter = 10) :
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y) :
        self.w_ = np.zeros(X.shape[1])
        self.cost_ = []

        for _ in range(n_iter) :
            output = self.net_input(X)
            errors = (y - output)
            self.w_[:1] =+ self.eta*X.T.dot(errors)
            self.w_[0] =+ self.eta*errors.sum()
            cost = (errors**2).sum()*self.eta/2
            self.cost_.append(cost)
        return self

    def net_input(self, X) :
        return np.dot(X, self.w_[:1] + self.w_[0])

    def predict(self, X, y) :
        return np.where(self.net_input(X) >= 0, -1, 1)
