import torch as th
import torch.nn as nn
from gradient_reversal import GradientReversal

class DragonNet(nn.Module):
    """
    Implementation of the DragonNet from "Adapting Neural Networks for the Estimation of Treatment Effects"
        available at https://arxiv.org/abs/1906.02120
    """

    def __init__(self, input_dim, hidden_dim=5, deconf_dim=2, outcome_dim=5, num_shared_layers=3, num_head_layers=3, alpha=1., reverse_grad=False, reverse_alph=0.5):
        """


        :param input_dim:       dimensionality of the inputs
        :param hidden_dim:      dimensionality of the hidden shared layers
        :param deconf_dim:      dimensionality of the shared layer outputs
        :param outcome_dim:     dimensionality of the hidden outcome layers
        :param num_shared_layers:   number of shared layers
        :param num_head_layers:     number of outcome head layers
        :param alpha:           strength of treatment prediction term in objective function
        :param reverse_grad:    apply gradient reversal on treatment prediction head
        :param reverse_alph:    strength of gradient reversal layer
        """
        super(DragonNet, self).__init__()

        self.name = "DragonNet"
        self.alpha = alpha
        self.shared_layers = self.create_shared(input_dim, hidden_dim, deconf_dim, num_layers=num_shared_layers)
        self.outcome_layers = self.create_heads(deconf_dim, outcome_dim, num_layers=num_head_layers)    # arr of [head0, head1]
        self.treatment_layer = self.create_treatment_head(deconf_dim, reverse_grad=reverse_grad, reverse_alph=reverse_alph)

        self.criterion_y = nn.MSELoss()
        self.criterion_t = nn.BCELoss()
        return

    def forward(self, x, t):
        """
        Put input features x and treatments t through the network to obtain corresponding predicted outcome values.

        :param x:   features
        :param t:   assigned binary (0/1) treatments
        :return:    predicted outcomes for treatments, obtained feature representations after the shared layers, propensity scores
        """
        res = th.zeros(len(t))
        repres = self.shared_layers(x)

        for treat in range(2):
            res[t == treat] += self.outcome_layers[treat](repres[t == treat]).squeeze(dim=1)

        t_pred = th.sigmoid(self.treatment_layer(repres)).squeeze(dim=1)

        return res, repres, t_pred

    def create_shared(self, input_dim, hidden_dim, deconf_dim, num_layers=3):
        """
        Creates the shared representation layers.
        Layers connected by ELU and a batchnormalisation is applied after last layer.

        :param input_dim:       dimensionality of the inputs
        :param hidden_dim:      dimensionality of the hidden shared layers
        :param deconf_dim:      dimensionality of the shared layer outputs
        :param num_layers:      number of shared layers
        :return:                shared representation layers
        """
        layers = nn.Sequential()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < (num_layers-1) else deconf_dim
            layers.append(nn.Linear(in_dim, out_dim))

            if i == (num_layers-1):
                layers.append(nn.BatchNorm1d(deconf_dim))
            else:
                layers.append(nn.ELU())

        return layers

    def create_heads(self, deconf_dim, outcome_dim, num_layers=3):
        """
        Creates the two outcomes heads to predict outcomes given treatment (1) or no treatment (0).
        Layers connected by ELU.

        :param deconf_dim:      dimensionality of the previous shared layer outputs
        :param outcome_dim:     dimensionality of the hidden outcome layers
        :param num_layers:      number of outcome head layers for both heads
        :return:                module list of outcome heads containing [head0, head1]
        """
        heads = []

        for _ in range(2):
            layers = nn.Sequential()

            for i in range(num_layers):
                in_dim = deconf_dim if i == 0 else outcome_dim
                out_dim = outcome_dim if i < (num_layers - 1) else 1
                layers.append(nn.Linear(in_dim, out_dim))

                if i < (num_layers - 1):
                    layers.append(nn.ELU())

            heads.append(layers)

        return nn.ModuleList(heads)

    def create_treatment_head(self, deconf_dim, reverse_grad=False, reverse_alph=0.5):
        """
        Creates treatment prediction head.

        :param deconf_dim:      dimensionality of the previous shared layer outputs
        :param reverse_grad:    flag for applying gradient reversal
        :param reverse_alph:    strength of gradient reversal
        :return:                treatment prediction head
        """
        treatment_layer = None

        if reverse_grad:
            treatment_layer = nn.Sequential(
                GradientReversal(alpha=reverse_alph),
                nn.Linear(in_features=deconf_dim, out_features=1)
            )
        else:
            treatment_layer = nn.Linear(in_features=deconf_dim, out_features=1)

        return treatment_layer

    def calc_loss(self, x, t, y):
        """
        Calculate loss based on outcome and treatment predictions.

        :param x:   input features
        :param t:   input binary treatments
        :param y:   expected outcomes
        :return:    total objective loss
        """
        outputs, repr, t_pred = self.forward(x, t)
        loss = self.criterion_y(outputs, y)
        loss += self.alpha * self.criterion_t(t_pred, t)

        return loss
