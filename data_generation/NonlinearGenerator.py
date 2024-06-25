import numpy as np

from scipy.stats import truncnorm


def default_fy(X):
    """
    The default outcome function used for representation learning experiments.

    :param X:   confounders (dim 4 assumed by default)
    :return:    Y0 without noise
    """
    Y0 = X[:, 1:3] ** 2 @ np.array([0.25, 0.5])
    Y0 += np.expand_dims(X[:, 3], axis=1) ** 3 @ np.array([0.3])
    Y0 += np.expand_dims(X[:, 0], axis=1) @ np.array([1])
    return Y0


def default_fy_prognostic(X):
    """
    The default outcome function used for experiments with prognostic factor.

    :param X:   confounders (dim 5 assumed by default, last dim for prognostic)
    :return:    Y0 without noise with prognostic feature
    """
    return default_fy(X) + np.expand_dims(X[:, 4], axis=1) ** 2 @ np.array([0.5])  # prognostic term


def default_f_tau_modif(X):
    """
    The default non-linear effect modification function used for experiments.

    :param X:   confounders (dim 4 assumed by default)
    :return:    heterogeneous treatment effects
    """
    return np.expand_dims(X[:, 0], axis=1) ** 2 @ np.array([0.5]) + np.expand_dims(X[:, 1], axis=1) @ np.array([1])


class NonlinearGenerator:
    """
    Data generator for experiments with representation learning methods.
    Optional prognostic and instrumental variables, and effect modification function.
    """

    def __init__(self, overlap_level=1, num_conf=4, num_prog=0, num_iv=0, f_y=default_fy, prob_t=0.5, y_sigma=0.5, tau=2., f_tau_modif=None):
        """
        Initialise datagenerator with specific settings.

        :param overlap_level:   shifted in mean of treated confounder distributions
        :param num_conf:    number of confounders
        :param num_prog:    number of prognostic variables
        :param num_iv:      number of instrumental variables
        :param f_y:     outcome function to use (inputs X of dimension num_conf + num_prog assumed)
        :param prob_t:  portion of population treated
        :param y_sigma:     variance of outcome noise
        :param tau:     constant treatment effect
        :param f_tau_modif:     treatment effect modification function based on confounders
        """
        self.overlap_level = overlap_level
        self.prob_t = prob_t

        self.num_iv = num_iv
        self.num_prog = num_prog
        self.num_conf = num_conf

        self.f_y = f_y
        self.y_sigma = y_sigma

        self.f_tau_modif = f_tau_modif
        self.tau = tau
        return

    def generate_confounders(self, num_samples, num_conf, T, mean_shift):
        """
        Generate standard normal confounders with treated distribution shifted to right by self.overlap_level.

        :param num_samples: number of samples
        :param num_conf:    number of confounders
        :param T:   treatment assignments
        :param mean_shift:  shift in treated distribution mean to decrease overlap
        :return:    confounders
        """
        Xc = np.random.normal(0, 1, size=(num_samples - np.sum(T), num_conf))
        Xt = np.random.normal(mean_shift, 1, size=(np.sum(T), num_conf))

        X = np.zeros((num_samples, num_conf))
        X[T == 1] += Xt
        X[T == 0] += Xc

        return X

    def get_data(self, num_samples=10000):
        """
        Sample data according to given settings.

        :param num_samples:     number of samples
        :return:    dict of outcomes
        """
        T = np.random.binomial(1, self.prob_t, num_samples)

        X = self.generate_confounders(num_samples, self.num_conf, T, self.overlap_level)
        if self.num_prog > 0:
            X_prog = self.generate_confounders(num_samples, self.num_prog, T, 0)
            X = np.concatenate((X, X_prog), axis=1)

        Y0 = self.f_y(X) + np.random.normal(0, self.y_sigma, size=num_samples)
        Y1 = Y0 + self.tau
        if self.f_tau_modif is not None:
            Y1 += self.f_tau_modif(X)
        Y = T * Y1 + (1 - T) * Y0

        if self.num_iv > 0:
            X_iv = self.generate_confounders(num_samples, self.num_iv, T, self.overlap_level)
            X = np.concatenate((X, X_iv), axis=1)

        return {'X': X, 'T': T, 'Y': Y, 'Y0': Y0, 'Y1': Y1}


if __name__ == '__main__':
    datagen = NonlinearGenerator()
    data = datagen.get_data(5)
    print(data)

    datagen = NonlinearGenerator(f_y=default_fy_prognostic, num_prog=1, num_iv=1, f_tau_modif=default_f_tau_modif)
    data = datagen.get_data(10000)
    print(f"Approx. ATE: {np.mean(data['Y1'] - data['Y0'])}")
