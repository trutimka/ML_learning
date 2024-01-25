import numpy as np


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        median = np.median(x, axis=0)
        abs_deviation = np.abs(x - median)
        mean_abs_deviation = np.mean(abs_deviation, axis=0)
        return mean_abs_deviation
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        mean_abs_deviation = self.mean_abs_deviation_from_median(features)
        self.loc = np.median(features, axis=0)
        self.scale = mean_abs_deviation
        ####

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        log_pdf = -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        return log_pdf
        ####

    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
