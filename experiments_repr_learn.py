import os
import json
import time
import numpy as np
import torch as th
import torch.nn as nn

from tqdm import tqdm
from models.tarnet import Tarnet
from covar_metric import covar_metric
from models.dragonnet import DragonNet
from models.IPM import wasserstein_loss
from data_generation.NonlinearGenerator import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression


def train(train_loader, model, optimizer):
    avg_loss = 0
    for i, data_tr in enumerate(train_loader):
        optimizer.zero_grad()

        x, t, y = data_tr
        loss = model.calc_loss(x, t, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss

    return avg_loss / len(train_loader)


def run(model, data_train, data_test, ate=2., lamb=1., epochs=25, batch_size=512, tarnet=True, eps=0.001, lr=1e-1):
    data_scaler = StandardScaler()
    X_tr = data_scaler.fit_transform(data_train['X'])
    X_test = data_scaler.transform(data_test['X'])

    tensors_train = th.Tensor(X_tr), th.Tensor(data_train['T']), th.Tensor(data_train['Y'])
    train_set = TensorDataset(*tensors_train)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    tensors_test = th.Tensor(X_test), th.Tensor(data_test['T']), th.Tensor(data_test['Y']), th.Tensor(
        data_test['Y0']), th.Tensor(data_test['Y1'])
    test_set = TensorDataset(*tensors_test)
    test_loader = DataLoader(test_set, batch_size=len(data_test['T']))

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    val_mses = []
    val_ipms = []
    val_biases = []
    ipw_biases = []
    ipw_prog_biases = []
    aipw_biases = []
    aipw_prog_biases = []
    covars = []
    covars_eq_avg = []
    covars_prop = []
    covars_prop_overlap = []

    for i in tqdm(range(epochs)):
        train_loss = train(train_loader, model, optimizer)
        train_losses.append(train_loss.detach().numpy())

        if i == epochs - 1:
            with th.no_grad():
                val_loss = 0
                val_mse = 0
                val_ipm = 0
                val_bias = 0
                ipw_bias = 0
                ipw_prog_bias = 0
                aipw_bias = 0
                aipw_prog_bias = 0
                avg_covar = 0
                avg_covar_eq_avg = 0
                avg_covar_prop = 0
                avg_covar_prop_overlap = 0

                for data in test_loader:
                    x, t, y, y0, y1 = data
                    t1 = th.ones_like(t)
                    t0 = th.zeros_like(t)
                    if tarnet:
                        outputs, repres = model(x, t)
                    else:   # DragonNet
                        outputs, repres, _ = model(x, t)

                    val_loss += model.calc_loss(x, t, y)
                    val_mse += criterion(outputs, y)
                    val_ipm += wasserstein_loss(repres, t, lamb=lamb)

                    # Normalise the obtained representation
                    repres_unnorm = repres.numpy()
                    data_scaler = StandardScaler()
                    repres = data_scaler.fit_transform(repres_unnorm)

                    out_cop = outputs.numpy()
                    y_cop = y.numpy()
                    y0 = y0.numpy()
                    y1 = y1.numpy()
                    t = t.numpy()

                    if tarnet:
                        out1, _ = model(x, t1)
                        out0, _ = model(x, t0)
                    else:
                        out1, _, _ = model(x, t1)
                        out0, _, _ = model(x, t0)

                    ate_est = th.mean(out1 - out0)
                    val_bias += ate_est - ate

                    out1 = out1.numpy()
                    out0 = out0.numpy()

                    e_x = LogisticRegression()
                    e_x.fit(repres, t)
                    props = e_x.predict_proba(repres)[:, 1]

                    ipw_bias += np.mean((t / (props + eps) - (1 - t) / (1 - props + eps)) * y_cop) - ate
                    aipw_bias += np.mean(out1) - np.mean(out0) + np.mean(
                        (t / (props + eps) - (1 - t) / (1 - props + eps)) * (y_cop - out_cop)) - ate

                    e_prog = LogisticRegression()
                    e_prog.fit(out0.reshape(-1, 1), t)
                    props_prog = e_prog.predict_proba(out0.reshape(-1, 1))[:, 1]

                    ipw_prog_bias += np.mean((t / (props_prog + eps) - (1 - t) / (1 - props_prog + eps)) * y_cop) - ate
                    aipw_prog_bias += np.mean(out1) - np.mean(out0) + np.mean(
                        (t / (props_prog + eps) - (1 - t) / (1 - props_prog + eps)) * (y_cop - out_cop)) - ate

                    avg_covar_prop += covar_metric(repres, props, t, y0, y1)

                val_losses.append(val_loss.item() / len(test_loader))
                val_mses.append(val_mse.item() / len(test_loader))
                val_ipms.append(val_ipm.item() / len(test_loader))
                val_biases.append(val_bias.item() / len(test_loader))
                ipw_biases.append(ipw_bias / len(test_loader))
                ipw_prog_biases.append(ipw_prog_bias / len(test_loader))
                aipw_biases.append(aipw_bias / len(test_loader))
                aipw_prog_biases.append(aipw_prog_bias / len(test_loader))

                covars.append(avg_covar / len(test_loader))
                covars_eq_avg.append(avg_covar_eq_avg / len(test_loader))
                covars_prop.append(avg_covar_prop / len(test_loader))
                covars_prop_overlap.append(avg_covar_prop_overlap / len(test_loader))

    return {"val_losses": val_losses, "val_mses": val_mses, "val_ipms": val_ipms, "val_biases": val_biases,
            "covars": covars, "covars_eq_avg": covars_eq_avg, "covars_prop": covars_prop,
            "covars_prop_overlap": covars_prop_overlap, "ipw_biases": ipw_biases, "ipw_prog_biases": ipw_prog_biases,
            "aipw_biases": aipw_biases, "aipw_prog_biases": aipw_prog_biases}


def nonlinear_experiment(num_conf=4, num_prog=0, num_iv=0, f_y=default_fy, ate=2., f_tau_modif=None,
                         num_shared_layers=2, num_head_layers=1, dir_name="results/non-linear",
                         lamb=1., epochs=25, runs=25):
    seed = 10
    np.random.seed(seed=seed)
    th.manual_seed(seed)

    levels = [0.0, 0.5, 1.0, 1.5, 2.]
    alphas = [0., 0.25, 0.5, 0.75, 1.]

    start_time = time.time()
    print(f"\n\n\n%%%%%%%%%% Starting Experiment %%%%%%%%%%\n\n\n")
    for i in levels:
        print(f"\n%%%%%%%%%% Overlap value currently at: {i} %%%%%%%%%%\n")

        dir_name = f"{dir_name}/non-linear_lvl_{i}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        res = {}
        res_dnet = {}
        for alpha in alphas:
            res[alpha] = []
            res_dnet[alpha] = []

        generator = NonlinearGenerator(overlap_level=i, num_conf=num_conf, num_prog=num_prog, num_iv=num_iv, f_y=f_y,
                                       tau=ate, f_tau_modif=f_tau_modif)
        ate_true = ate
        if f_tau_modif is not None:
            data_test = generator.get_data(num_samples=100000)
            ate_true = np.mean(data_test['Y1'] - data_test['Y0'])
        data_test = generator.get_data(num_samples=10000)

        for j in range(runs):
            data_train = generator.get_data(num_samples=10000)
            for alpha in alphas:
                tarnet = True
                imb_model = Tarnet(data_train['X'].shape[1], alpha=alpha, lamb=lamb,
                                   num_shared_layers=num_shared_layers, num_head_layers=num_head_layers,
                                   IPM=True)
                res[alpha].append(run(imb_model, data_train, data_test, lamb=lamb, ate=ate_true, tarnet=tarnet, epochs=epochs))

                tarnet = False
                imb_model = DragonNet(data_train['X'].shape[1], alpha=alpha, num_shared_layers=num_shared_layers, num_head_layers=num_head_layers)
                res_dnet[alpha].append(run(imb_model, data_train, data_test, lamb=lamb, ate=ate_true, tarnet=tarnet, epochs=epochs))

        for alpha in alphas:
            subdir = f"{dir_name}/non-linear_imb_level_{i}_tarnet_{alpha}_lamb{lamb}/results.json"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            with open(subdir, "w") as f:
                f.write(json.dumps(res[alpha], indent=4))

            subdir = f"{dir_name}/non-linear_imb_level_{i}_dnet_layer_{alpha}_lamb{lamb}/results.json"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            with open(subdir, "w") as f:
                f.write(json.dumps(res_dnet[alpha], indent=4))

    print(f"\n%%%%% Time taken: {(time.time() - start_time) / 60} mins %%%%%")
    return


if __name__ == '__main__':
    nonlinear_experiment()
    nonlinear_experiment(f_tau_modif=default_f_tau_modif, dir_name="results/non-linear_effect_modif")
    nonlinear_experiment(num_prog=1, f_y=default_fy_prognostic, dir_name="results/non-linear_prognostic")
    nonlinear_experiment(num_iv=1, dir_name="results/non-linear_iv")
