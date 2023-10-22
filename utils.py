import torch
import torch.nn as nn


def init_weights(weight=0.01, bias=0.01):
    def function(model): 
        if isinstance(model, nn.Linear):
            model.weight.data.fill_(weight)
            model.bias.data.fill_(bias)
    return function


def euclidean_distance(model1, model2):
    params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.view(-1) for p in model2.parameters()])
    return torch.norm(params1 - params2).item()
