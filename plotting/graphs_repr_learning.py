import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

keys = {
    "val_losses": "Loss total",
    "val_ipms": "IPM value",
    "val_mses": "MSE's",
    "val_biases": "Bias in ATE",
    "covars_prop": "Confounding bias in representation",
    "ipw_biases": "IPW bias",
    "ipw_prog_biases": "IPW prognostic scores bias",
    "aipw_biases": "AIPW bias",
    "aipw_prog_biases": "AIPW prognostic scores bias"
}

labels = {
    "val_losses": "Loss",
    "val_ipms": "IPM",
    "val_mses": "MSE",
    "val_biases": "Bias",
    "covars_prop": "Bias",
    "ipw_biases": "Bias",
    "ipw_prog_biases": "Bias",
    "aipw_biases": "Bias",
    "aipw_prog_biases": "Bias"
}

model_names = {
    "tarnet": "CFR",
    "tarnet_no_ipm": "TARNet without IPM",
    "dnet": "DragonNet",
    "dnet_reversed": "DragonNet with gradient reversal"
}

prefix_names = {
    "": "for non-overlapping linear case",
    "non-": "for non-overlapping non-linear case",
    "full_": "for fully overlapping linear case",
    "full_non-": "for fully overlapping non-linear case"
}

colors = {
    "0.0": "tab:blue",
    "0.25": "tab:orange",
    "0.5": "tab:green",
    "0.75": "tab:red",
    "1.0": "tab:purple"
}
markers = {
    "0.0": "s",
    "0.25": "o",
    "0.5": "*",
    "0.75": "^",
    "1.0": "D"
}

models = ["tarnet", "dnet"]
levels = [0.0, 0.5, 1.0, 1.5, 2.]
alphas = [0., 0.25, 0.5, 0.75, 1.]
lamb = 1.


def plot_repr(dir, ylims=None):
    """
    Plots results for representation learning models.

    :param dir:     directory with the results
    :param ylims:   optional array with y-axis limits [lower, upper]
    :return:    -
    """
    for model in models:
        values = {}

        for key, val in keys.items():
            values[key] = {}
            results = {}
            vars = {}

            fig, ax = plt.subplots()
            ax.set(xlabel="Distance between means of treated and control distributions", ylabel=f"{labels[key]}",
                   title=f"{model_names[model]}\n{keys[key]}")

            for a in alphas:
                values[key][f"{a}"] = {}
                results[f"{a}"] = []
                vars[f"{a}"] = []
                runs = 0
                for i in levels:
                    values[key][f"{a}"][f"{i}"] = []

                    with open(f"{dir}/non-linear_lvl_{i}/non-linear_imb_level_{i}_{model}_2layer_{a}_lamb{lamb}/results.json") as f:# with open(f"{dir}/{dir}_non-linear_lvl_{i}/non-linear_imb_level_{i}_{model}_2layer_{a}_lamb{lamb}/results.json") as f:
                        res = json.load(f)
                        runs = len(res)

                        for entry in res:
                            values[key][f"{a}"][f"{i}"].append(entry[key][-1])
                    results[f"{a}"].append(np.mean(values[key][f"{a}"][f"{i}"]))
                    vars[f"{a}"].append(np.var(values[key][f"{a}"][f"{i}"]))

                runs = np.sqrt(runs)
                b = np.array(results[f"{a}"])
                v = np.array(vars[f"{a}"])
                plt.plot(levels, b, label=r'$ \alpha $' + f' = {a}', color=colors[f"{a}"], marker=markers[f"{a}"])
                plt.fill_between(levels, b - v / runs, b + v / runs, color=colors[f"{a}"], alpha=0.15)

            if ylims:
                plt.ylim(ylims[0], ylims[1])

            plt.legend(loc='upper left')
            fig.savefig(f"{dir}/{model}_{keys[key]}")
            plt.close(fig)


if __name__ == '__main__':
    dir_name = "results/non-linear"
    plot_repr(dir_name)
