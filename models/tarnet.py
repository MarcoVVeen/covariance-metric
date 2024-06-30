import torch as th
import torch.nn as nn

from models.IPM import wasserstein_loss


class Tarnet(nn.Module):
    """
    Implementation of the TARNet/CFR models as described in the paper "Estimating individual treatment effect: generalization bounds and algorithms"
        available at https://arxiv.org/abs/1606.03976

    The Wasserstein metric is used for the balancing through the IPM objective.
    """

    def __init__(self, input_dim, hidden_dim=5, deconf_dim=2, outcome_dim=5, num_shared_layers=3, num_head_layers=3, alpha=0.01, lamb=1., IPM=True):
        """
        Initialises the TARNet model. An alpha of 0 results in the TARNet, while a non-zero alpha results in the CFR
        using Wasserstein distances.

        :param input_dim:       dimensionality of the inputs
        :param hidden_dim:      dimensionality of the hidden shared layers
        :param deconf_dim:      dimensionality of the shared layer outputs
        :param outcome_dim:     dimensionality of the hidden outcome layers
        :param num_shared_layers:   number of shared layers
        :param num_head_layers:     number of outcome head layers
        :param alpha:           strength of IPM term in objective function
        :param lamb:            input parameter for Wasserstein algorithm based on Sinkhorn distances
        :param IPM:             flag for using IPM objective or not. same effect as setting alpha to 0, but also skips
                                    unnecessary Wasserstein approximation if explicitly set to False.
        """
        super(Tarnet, self).__init__()

        self.name = "TARNet"
        self.shared_layers = self.create_shared(input_dim, hidden_dim, deconf_dim, num_layers=num_shared_layers)
        self.outcome_layers = self.create_heads(deconf_dim, outcome_dim, num_layers=num_head_layers)    # arr of [head0, head1]

        self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.lamb = lamb
        self.IPM = IPM
        return

    def forward(self, x, t):
        """
        Put input features x and treatments t through the network to obtain corresponding predicted outcome values.

        :param x:   features
        :param t:   assigned binary (0/1) treatments
        :return:    predicted outcomes for the given treatments and the obtained feature representations after the shared layers
        """
        res = th.zeros(len(t))
        repres = self.shared_layers(x)

        for treat in range(2):
            res[t == treat] += self.outcome_layers[treat](repres[t == treat]).squeeze(dim=1)

        return res, repres

    def create_shared(self, input_dim, hidden_dim, deconf_dim, num_layers=3):
        """
        Creates the shared representation layers.
        Layers connected by ELU and a batchnormalisation is applied after last layer as described in the original paper.

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

            # Last layer is followed by a batch normalisation; otherwise connect layers with ELU.
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

    def calc_loss(self, x, t, y):
        """
        Calculate the CFR loss consisting of the MSE and IPM.

        :param x:   input features
        :param t:   input binary treatments
        :param y:   expected outcomes
        :return:    total objective loss
        """
        outputs, repr = self.forward(x, t)
        loss = self.criterion(outputs, y)

        if self.IPM and self.alpha > 0.0001:
            wass = wasserstein_loss(repr, t, lamb=self.lamb)
            loss += self.alpha * wass

        return loss
