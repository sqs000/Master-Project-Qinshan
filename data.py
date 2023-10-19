import cocoex as ex
import numpy as np
import torch


class data_generator:

    def __init__(self, suite_name, function, dimension, instance):        
        self.dimension = dimension
        self.suite = ex.Suite(suite_name=suite_name, suite_instance='', suite_options='')
        self.problem = self.suite.get_problem_by_function_dimension_instance(function=function, dimension=dimension, instance=instance)

    def generate(self, data_size, x_min=-5, x_max=5):
        x = []
        y = []
        for i in range(data_size):
            input = np.random.uniform(x_min, x_max, self.dimension)
            output = self.problem(input)
            x.append(input)
            y.append([output])        
        x_tensor = torch.FloatTensor(np.array(x))
        y_tensor = torch.FloatTensor(np.array(y))
        return x_tensor, y_tensor
            
        
