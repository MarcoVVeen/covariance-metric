import os
import json
import time
import numpy as np

from covar_metric import covar_metric
from data_generation.LinearGenerator import LinearGenerator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def experiment_base(data_train, data_test, ate, use_predicted=False):
    """
    Base experiment loop.

    :param data_train:  train set
    :param data_test:   test set for evaluation
    :param ate:     true ate
    :param use_predicted:   use predicted values instead of potential outcomes
    :return:    dict with ate bias and measured confounding bias in representation (= 2 obs. confounders)
    """
    run_res = {}

    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(data_train['X'])
    x_test = data_scaler.transform(data_test['X'])

    t = data_train['T']
    y = data_train['Y']

    m0 = LinearRegression()
    m0.fit(x[t == 0], y[t == 0])
    y0_pred = m0.predict(x_test)

    m1 = LinearRegression()
    m1.fit(x[t == 1], y[t == 1])
    y1_pred = m1.predict(x_test)

    ate_bias = np.mean(y1_pred - y0_pred) - ate
    run_res['ate_bias'] = ate_bias

    t_test = data_test['T']
    y0_test = data_test['Y0']
    y1_test = data_test['Y1']
    repres = x_test

    # Fit reduced propensity scores for covariance metric denominators
    e_x = LogisticRegression()
    e_x.fit(x, t)
    reduced_prop = e_x.predict_proba(repres)[:, 1]

    run_res['covars'] = covar_metric(repres, reduced_prop, t_test, y0_test, y1_test)

    # fit models and predict y and propensity to use instead of potential outcomes and t
    # has to be done on unconfounded covariates, so include U (X was our fake "representation" which lost U info)
    if use_predicted:
        data_scaler = StandardScaler()
        u = data_scaler.fit_transform(data_train['U'])
        u_test = data_scaler.transform(data_test['U'])

        x_tot = np.concatenate((x, u), axis=1)
        x_test_tot = np.concatenate((x_test, u_test), axis=1)

        m0_tot = LinearRegression()
        m0_tot.fit(x_tot[t == 0], y[t == 0])
        y0_tot = m0_tot.predict(x_test_tot)

        m1_tot = LinearRegression()
        m1_tot.fit(x_tot[t == 1], y[t == 1])
        y1_tot = m1_tot.predict(x_test_tot)

        e_x_tot = LogisticRegression()
        e_x_tot.fit(x_tot, t)
        props_tot = e_x_tot.predict_proba(x_test_tot)[:, 1]

        run_res['covars_est'] = covar_metric(repres, reduced_prop, props_tot, y0_tot, y1_tot)

    return run_res


def hidden_confounder_experiment(hidden_coeffs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.], ate=2, overlap="incomplete", use_predicted=False,
                                 train_samples=10000, test_samples=10000, seed=10, runs=25, dir_name="results/hidden_linear"):
    """
    Base hidden confounder experiment for increasing outcome strength of hidden confounder.

    :param hidden_coeffs:   list of increasing coefficients for hidden confounder
    :param ate:     true ate
    :param runs:    num runs
    :param dir_name:    results directory
    :return:    -
    """
    np.random.seed(seed=seed)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print(f"\nSaving in dir {dir_name}\n")

    for i in hidden_coeffs:
        print(f"Current hidden confounder strength: {i}")
        res = []

        for _ in range(runs):
            generator = LinearGenerator(beta_hidden=[i], tau=ate, overlap=overlap)
            data_train = generator.get_data(num_samples=train_samples)
            data_test = generator.get_data(num_samples=test_samples)

            res.append(experiment_base(data_train, data_test, ate, use_predicted=use_predicted))

        if not os.path.exists(f"{dir_name}/coeff_{i}"):
            os.makedirs(f"{dir_name}/coeff_{i}")

        with open(f"{dir_name}/coeff_{i}/results.json", "w") as f:
            f.write(json.dumps(res, indent=4))
    return


if __name__ == '__main__':
    hidden_confounder_experiment()
    # hidden_confounder_experiment(test_samples=100000, dir_name="results/hidden_linear_100k")

    hidden_confounder_experiment(overlap="complete", dir_name="results/hidden_linear_complete")

    hidden_confounder_experiment(use_predicted=True, dir_name="results/hidden_linear_predicted")
