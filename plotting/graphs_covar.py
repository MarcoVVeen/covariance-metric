import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

keys = {
    "ate_bias": "Bias in ATE estimates",
    "covars": "Confounding bias in representation",
    "covars_est": "Bias using predictions"
}

colors = {
    "ate_bias": "r",
    "covars": "b",
    "covars_est": 'g'
}
markers = {
    "ate_bias": "s",
    "covars": "o",
    "covars_est": "x"
}

hidden_conf_strength = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]


def plot_covar(dir, ylims=None, plot_title=None):
    """
    Plots results for covariance metric evaluation experiment.

    :param dir:     directory with the results
    :param ylims:   optional array with y-axis limits [lower, upper]
    :param plot_title:  optional other title for plot
    :return:    -
    """
    values = {}
    biases = {}
    vars = {}
    for key, val in keys.items():
        values[key] = {}
        biases[key] = []
        vars[key] = []
        for i in hidden_conf_strength:
            values[key][f"{i}"] = []

    runs = 0
    for i in hidden_conf_strength:
        with open(f"{dir}/coeff_{i}/results.json") as f:
            res = json.load(f)
            runs = len(res)

            for entry in res:
                for key, val in keys.items():
                    values[key][f"{i}"].append(entry[key])
    fig, ax = plt.subplots()
    ax.set(xlabel="Hidden confounder strength in outcomes Y", ylabel="Bias",
           title=f"Hidden Confounder with incomplete overlap" if plot_title is None else plot_title)
    runs = np.sqrt(runs)
    for key, val in values.items():
        for i in hidden_conf_strength:
            biases[key].append(np.mean(values[key][f"{i}"]))
            vars[key].append(np.var(values[key][f"{i}"]))

        b = np.array(biases[key])
        v = np.array(vars[key])
        plt.plot(hidden_conf_strength, b, label=f'{keys[key]}', color=colors[key], marker=markers[key])
        plt.fill_between(hidden_conf_strength, b - v / runs, b + v / runs, color=colors[key], alpha=0.15)

    if ylims:
        plt.ylim(ylims[0], ylims[1])
    
    plt.legend()
    fig.savefig(f"{dir}/covar_hidden_conf")
    plt.close(fig)


if __name__ == '__main__':
    dir_name = "results/hidden_linear"
    plot_covar(dir_name)
