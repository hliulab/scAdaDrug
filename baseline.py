import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function

device = 'cuda'

class AdversarialNetwork(nn.Module):
    def __init__(self, device, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size, device=device)
        self.bn = nn.BatchNorm1d(hidden_size, device=device)
        self.relu = nn.ReLU()
        self.ad_layer2 = nn.Linear(hidden_size, 1, device=device)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.ad_layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        y = self.ad_layer2(x)
        return torch.sigmoid(y)


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse.apply(x)


# Define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, device, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, encoding_dim, device=device)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, input_dim, device=device)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Define predictor model
class Predictor(nn.Module):
    def __init__(self, device, encoding_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(encoding_dim, 64, device=device)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 16, device=device)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1, device=device)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = nn.Sigmoid()(self.fc3(x))
        return x



class Model(nn.Module):
    def __init__(self, device, ae_indim, embedding_dim):
        super(Model, self).__init__()
        self.autoencoder = Autoencoder(device=device, input_dim=ae_indim, encoding_dim=embedding_dim)
        # self.predictor = Predictor(device=device, input_dim=pre_indim, hidden_dim=pre_hiddim, output_dim=pre_outdim)
        self.predictor = Predictor(device=device, encoding_dim=embedding_dim)
        self.domain_classifier = AdversarialNetwork(device=device, in_feature=embedding_dim, hidden_size=64)

        # self.mlp = MLP(device=device, input_size=embedding_dim, output_size=embedding_dim)


    def forward(self, sx, tx):
        s_decoded, s_encoded = self.autoencoder(sx)
        t_decoded, t_encoded = self.autoencoder(tx)

        s_pred = self.predictor(s_encoded)
        t_pred = self.predictor(t_encoded)

        st_encoded = torch.cat((s_encoded, t_encoded), dim=0)
        domain_pre = self.domain_classifier(st_encoded)

        return s_encoded, s_decoded, s_pred, t_encoded, t_decoded, t_pred, domain_pre
