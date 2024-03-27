import cocoex as ex
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class data_generator:
    def __init__(self, suite_name, function, dimension, instance, device=torch.device("cpu")):        
        self.dimension = dimension
        self.suite = ex.Suite(suite_name=suite_name, suite_instance='', suite_options='')
        self.problem = self.suite.get_problem_by_function_dimension_instance(function=function, dimension=dimension, instance=instance)
        self.device = device

    def generate(self, data_size, x_min=-5, x_max=5):
        x = []
        y = []
        for i in range(data_size):
            input = np.random.uniform(x_min, x_max, self.dimension)
            output = self.problem(input)
            x.append(input)
            y.append([output])
        # standardize the data
        scaler = StandardScaler()
        x_tensor = torch.tensor(scaler.fit_transform(x, y), dtype=torch.float32, device=self.device)        
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor


class ForestDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        

def getForestCoverTypeDataset(file_path=None):
    # Define the column names based on the provided information
    column_names = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
        'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
        'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
        'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
        'Cover_Type'
    ]

    # Read the data into a pandas DataFrame
    forest_data = pd.read_csv(file_path, header=None, names=column_names)
    # Split the data into features (X) and labels (y)
    X = forest_data.drop('Cover_Type', axis=1).values
    y = forest_data['Cover_Type'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train - 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test - 1)
    train_dataset = ForestDataset(X_train_tensor, y_train_tensor)
    test_dataset = ForestDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
