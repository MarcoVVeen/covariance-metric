import numpy as np

from scipy.stats import truncnorm


class LinearGenerator:
    """
    Data generator used for experiments for evaluating the covariance metric.
    Outcomes are linear function of observed and hidden confounders.
    Optional additional squared effect modification for testing misspecification.
    """

    def __init__(self, beta0=0, beta=[0.75, 1], beta_hidden=[1], overlap="incomplete", overlap_level=1., prob_t=0.5, y_sigma=0.5, tau=2., tau_modif=[0, 0]):
        """
        Initialise overlap setting of confounders, and coefficients of the confounders and effect modification.

        :param beta0:   constant
        :param beta:    coefficient observed confounders. #confounders generated according to #coefficients
        :param beta_hidden:     coefficient hidden confounders. #confounders generated according to #coefficients
        :param overlap:     "incomplete" for structural overlap violations, "complete" for full, but low overlap
        :param overlap_level:   difference between distr. means when "incomplete", variance in truncated normals for "complete"
        :param prob_t:  portion of population receiving treatment
        :param y_sigma:     noise in outcomes
        :param tau:     constant treatment effect
        :param tau_modif:   squared effect modification coefficients for observed confounders
        """
        self.beta0 = beta0
        self.beta = np.array(beta)
        self.beta_hidden = np.array(beta_hidden)

        self.tau = tau
        self.tau_modif = np.array(tau_modif)

        self.prob_t = prob_t
        self.y_sigma = y_sigma

        self.generate_confounders = self.generate_confounders_incomplete if overlap=="incomplete" else self.generate_confounders_complete
        self.overlap_level = overlap_level
        return

    def generate_confounders_complete(self, num_samples, num_conf, T):
        """
        Generate truncated normal confounders with self.overlap_level being the variances, which control the overlap.
            overlap_level = 1.5     relatively good overlap entire interval
            overlap_level = 0.75    low overlap at tails
            overlap_level = 0.5     structural lack of overlap at extremes of interval

        :param num_samples: number of samples
        :param num_conf:    number of confounders
        :param T:   treatment assignments
        :return:    confounders
        """
        X = np.zeros((num_samples, num_conf))
        scale = self.overlap_level

        # all values truncated between 0 and 2. Control centered at 0, treated at 2.
        a = 0
        b = 2

        loc = 0
        a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
        rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
        for i in range(num_conf):
            X[T == 0, i] += rv.rvs(size=num_samples - np.sum(T))

        loc = 2
        a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
        rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
        for i in range(num_conf):
            X[T == 1, i] += rv.rvs(size=np.sum(T))

        return X

    def generate_confounders_incomplete(self, num_samples, num_conf, T):
        """
        Generate standard normal confounders with treated distribution shifted to right by self.overlap_level.

        :param num_samples: number of samples
        :param num_conf:    number of confounders
        :param T:   treatment assignments
        :return:    confounders
        """
        Xc = np.random.normal(0, 1, size=(num_samples - np.sum(T), num_conf))
        Xt = np.random.normal(self.overlap_level, 1, size=(np.sum(T), num_conf))    # overlap_level is shift in means

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

        X = self.generate_confounders(num_samples, len(self.beta), T)
        X_hidden = self.generate_confounders(num_samples, len(self.beta_hidden), T)

        Y0 = X @ self.beta + np.random.normal(0, self.y_sigma, size=num_samples)
        Y0 += X_hidden @ self.beta_hidden

        Y1 = Y0 + self.tau
        # Add sqr effect modifications if present
        if np.any(self.tau_modif):
            Y1 += X**2 @ self.tau_modif

        Y = T * Y1 + (1-T) * Y0
        return {'X': X, 'T': T, 'Y': Y, 'Y0': Y0, 'Y1': Y1, 'U': X_hidden}


if __name__ == '__main__':
    datagen = LinearGenerator()
    data = datagen.get_data(5)
    print(data)

    datagen = LinearGenerator(overlap="complete")
    data = datagen.get_data(5)
    print(data)

    datagen = LinearGenerator(tau_modif=[0.5, 0])
    data = datagen.get_data(10000)
    print(f"Approx. ATE: {np.mean(data['Y1'] - data['Y0'])}")
