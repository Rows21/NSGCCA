import torch
import torch.nn as nn

from loss_objectives import GCCA_loss, new_loss

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layer1 = nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1])
                nn.init.kaiming_uniform_(layer1.weight, nonlinearity='relu')
                nn.init.zeros_(layer1.bias)
                layer = nn.Sequential(
                    layer1,
                    #nn.Sigmoid(), 
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                )
                #nn.init.xavier_uniform_(layer.linear.weight)
                #nn.init.zeros_(layer.linear.bias)
                layers.append(layer)
            else:
                layer1 = nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1])
                nn.init.xavier_uniform_(layer1.weight)
                nn.init.zeros_(layer1.bias)
                layer = nn.Sequential(
                    layer1,
                    #nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                )
                layers.append(layer)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepGCCA(nn.Module):
    def __init__(self, layer_sizes_list, input_size_list, outdim_size, use_all_singular_values=False, device=torch.device('cpu')):
        super(DeepGCCA, self).__init__()
        self.model_list = []
        for i in range(len(layer_sizes_list)):
            self.model_list.append(MlpNet(layer_sizes_list[i], input_size_list[i]).double())
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = new_loss #GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x)) 

        return output_list
