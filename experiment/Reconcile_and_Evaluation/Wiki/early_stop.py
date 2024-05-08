import numpy as np

class EarlyStopping:
    '''
    class to define whether to early stop
    '''
    def __init__(self, patience=100, delta=5000000):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.Inf
        self.wait = 0

    def __call__(self, val_loss):
        """
        call this
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.wait = 0

        else:
            self.wait += 1

        if self.wait >= self.patience:
            return True
        return False


