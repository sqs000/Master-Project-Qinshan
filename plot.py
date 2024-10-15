import os
import numpy as np
import matplotlib.pyplot as plt


# Function to load loss data from npy files
def load_losses(folder_path, max_iterations=None):
    if max_iterations:
        losses = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if "losses" in file:
                    loss_file = os.path.join(root, file)
                    loss_data = np.load(loss_file)
                    period = len(loss_data) // max_iterations
                    # uncomment the following line when plotting for 
                    # SGD_Adam_GA_GA_sharing_GA_dynamic_Fs(), SGD_Adam_GA_GA_sharing_GA_dynamic_F3(), GA_GA_sharing_GA_dynamic_no_crossover()
                    # loss_data = loss_data[period-1::period][1:]
                    # uncomment the following line when plotting for 
                    # GA_SGD_SGD_Adam(), GA_SGD_sharing_SGD(), GA_SGD_dynamic_SGD()
                    loss_data = loss_data[period-1::period]
                    losses.append(loss_data)
        return np.array(losses)
    else:
        losses = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if "losses" in file:
                    loss_file = os.path.join(root, file)
                    loss_data = np.load(loss_file)
                    losses.append(loss_data)
        return np.array(losses)

# Function to plot average loss curves
def plot_loss_curves(losses, title, labels, x_range=None):
    # tab10 or tab20 colormap for plotting
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.tab10.colors))
    for i, loss in enumerate(losses):
        mean_loss = np.mean(loss, axis=0)
        std_loss = np.std(loss, axis=0)
        if x_range:
            plt.plot(x_range, mean_loss, label=labels[i])
            plt.fill_between(x_range, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        else:
            plt.plot(mean_loss, label=labels[i])
            plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Number of evaluations (×5000000)', fontsize=14)
    plt.ylabel('MSE Loss (log scale)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()


# Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for all functions
def SGD_Adam_GA_GA_sharing_GA_dynamic_Fs():
    functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
    algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
    settings = ['param', 'layer', 'node']
    
    max_iterations = 100  # to compare SGD/Adam with GAs, sample 100 SGD/Adam losses 
    
    for func in functions:
        func_path = os.path.join('results3', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            alg_losses = []
            if alg == 'SGD' or alg == 'Adam':
                alg_losses.append(load_losses(alg_path, max_iterations))
                labels.append(alg)
            else:
                for setting in settings:
                    setting_path = os.path.join(alg_path, setting)
                    alg_losses.append(load_losses(setting_path))
                    labels.append(f'{alg} ({setting})')
            func_losses.extend(alg_losses)
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels, x_range=range(2, 101))

# Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for F3
def SGD_Adam_GA_GA_sharing_GA_dynamic_F3():
    algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
    settings = ['param', 'layer', 'node']
    
    max_iterations = 400  # to compare SGD/Adam with GAs, sample 100 SGD/Adam losses 
    func_losses = []
    labels = []
    for alg in algorithms:
        alg_path = os.path.join('results_F3', alg)
        alg_losses = []
        if alg == 'SGD' or alg == 'Adam':
            alg_losses.append(load_losses(alg_path, max_iterations))
            labels.append(alg)
        else:
            for setting in settings:
                setting_path = os.path.join(alg_path, setting)
                alg_losses.append(load_losses(setting_path))
                labels.append(f'{alg} ({setting})')
        func_losses.extend(alg_losses)
    plt.tight_layout()
    plot_loss_curves(func_losses, "F3", labels, x_range=range(2, 401))

# Plotting the GA-GA_sharing-GA_dynamic(param/node/layer/none) plots
def GA_GA_sharing_GA_dynamic_no_crossover():
    functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
    algorithms = ['GA', 'GA_sharing', 'GA_dynamic']
    settings = ['param', 'layer', 'node', 'none']
        
    for func in functions:
        func_path = os.path.join('results_crossover_ablation', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            alg_losses = []
            for setting in settings:
                setting_path = os.path.join(alg_path, setting)
                alg_losses.append(load_losses(setting_path))
                labels.append(f'{alg} ({setting})')
            func_losses.extend(alg_losses)
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels, x_range=range(2, 101))

# Plotting the GA_SGD-SGD-Adam plots
def GA_SGD_SGD_Adam():
    functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
    algorithms = ['GA_SGD', 'SGD', 'Adam']
    
    max_iterations = 100  # to compare SGD/Adam with GA_SGD, sample 100 SGD/Adam losses
    for func in functions:
        func_path = os.path.join('results_GA_SGD', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            if alg == "GA_SGD":
                func_losses.append(load_losses(alg_path))
            else:
                func_losses.append(load_losses(alg_path, max_iterations))
            labels.append(f'{alg}')
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels, x_range=range(1, 101))

# Plotting the GA_SGD-SGD-Adam plots
def GA_SGD_sharing_SGD():
    functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
    algorithms = ['GA_SGD', 'SGD', 'R=1', 'R=5', 'R=10', 'R=20', 'R=50']
    
    max_iterations = 100  # to compare SGD/Adam with GA_SGD, sample 100 SGD/Adam losses
    for func in functions:
        func_path = os.path.join('results_GA_SGD_sharing', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            if alg == 'SGD':
                func_losses.append(load_losses(alg_path, max_iterations))
            else:
                func_losses.append(load_losses(alg_path))
            labels.append(f'{alg}')
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels, x_range=range(1, 101))

# Plotting the GA_SGD-SGD-Adam plots
def GA_SGD_dynamic_SGD():
    functions = ['F3']
    algorithms = ['GA_SGD', 'SGD', 'R=5', 'R=5 N=5', 'R=5 N=10', 'R=5 N=20', 'R=5 N=50', 'R=5 N=100']
    
    max_iterations = 100  # to compare SGD/Adam with GA_SGD, sample 100 SGD/Adam losses
    for func in functions:
        func_path = os.path.join('results_GA_SGD_dynamic', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            if alg == 'SGD':
                func_losses.append(load_losses(alg_path, max_iterations))
            else:
                func_losses.append(load_losses(alg_path))
            labels.append(f'{alg}')
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels, x_range=range(1, 101))


if __name__ == "__main__":
    # SGD_Adam_GA_GA_sharing_GA_dynamic_Fs()
    # SGD_Adam_GA_GA_sharing_GA_dynamic_F3()
    # GA_GA_sharing_GA_dynamic_no_crossover()
    # GA_SGD_SGD_Adam()
    # GA_SGD_sharing_SGD()
    GA_SGD_dynamic_SGD()
