# # Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for all functions
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load loss data from npy files
# def load_losses(folder_path, max_iterations=None):
#     if max_iterations:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     period = len(loss_data) // max_iterations
#                     loss_data = loss_data[::period][1:]
#                     losses.append(loss_data)
#         return np.array(losses)
#     else:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     losses.append(loss_data)
#         return np.array(losses)

# # Function to plot average loss curves
# def plot_loss_curves(losses, title, labels):
#     for i, loss in enumerate(losses):
#         mean_loss = np.mean(loss, axis=0)
#         std_loss = np.std(loss, axis=0)
#         plt.plot(mean_loss, label=labels[i])
#         plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
#     plt.title(title)
#     plt.yscale('log')
#     plt.xlabel('Number of evaluations (×5000000)')
#     plt.ylabel('MSE Loss (log scale)')
#     plt.legend()
#     plt.show()

# # Main function to traverse directories and plot loss curves
# def main():
#     functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
#     algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
#     settings = ['param', 'layer', 'node']
    
#     max_iterations = 100  # to compare SGD/Adam with GAs, sample 100 SGD/Adam losses 
    
#     for func in functions:
#         func_path = os.path.join('results3', func)
#         func_losses = []
#         labels = []
#         for alg in algorithms:
#             alg_path = os.path.join(func_path, alg)
#             alg_losses = []
#             if alg == 'SGD' or alg == 'Adam':
#                 alg_losses.append(load_losses(alg_path, max_iterations))
#                 labels.append(alg)
#             else:
#                 for setting in settings:
#                     setting_path = os.path.join(alg_path, setting)
#                     alg_losses.append(load_losses(setting_path))
#                     labels.append(f'{alg} ({setting})')
#             func_losses.extend(alg_losses)
#         plt.tight_layout()
#         plot_loss_curves(func_losses, func, labels)

# if __name__ == "__main__":
#     main()


# # Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for F3
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load loss data from npy files
# def load_losses(folder_path, max_iterations=None):
#     if max_iterations:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     period = len(loss_data) // max_iterations
#                     loss_data = loss_data[::period][1:]
#                     losses.append(loss_data)
#         return np.array(losses)
#     else:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     losses.append(loss_data)
#         return np.array(losses)

# # Function to plot average loss curves
# def plot_loss_curves(losses, title, labels):
#     for i, loss in enumerate(losses):
#         mean_loss = np.mean(loss, axis=0)
#         std_loss = np.std(loss, axis=0)
#         plt.plot(mean_loss, label=labels[i])
#         plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
#     plt.title(title)
#     plt.yscale('log')
#     plt.xlabel('Number of evaluations (×5000000)')
#     plt.ylabel('MSE Loss (log scale)')
#     plt.legend()
#     plt.show()

# # Main function to traverse directories and plot loss curves
# def main():
#     algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
#     settings = ['param', 'layer', 'node']
    
#     max_iterations = 400  # to compare SGD/Adam with GAs, sample 100 SGD/Adam losses 
#     func_losses = []
#     labels = []
#     for alg in algorithms:
#         alg_path = os.path.join('results_F3', alg)
#         alg_losses = []
#         if alg == 'SGD' or alg == 'Adam':
#             alg_losses.append(load_losses(alg_path, max_iterations))
#             labels.append(alg)
#         else:
#             for setting in settings:
#                 setting_path = os.path.join(alg_path, setting)
#                 alg_losses.append(load_losses(setting_path))
#                 labels.append(f'{alg} ({setting})')
#         func_losses.extend(alg_losses)
#     plt.tight_layout()
#     plot_loss_curves(func_losses, "F3", labels)

# if __name__ == "__main__":
#     main()


# # Plotting the GA-GA_sharing-GA_dynamic(param/node/layer/none) plots
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load loss data from npy files
# def load_losses(folder_path, max_iterations=None):
#     if max_iterations:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     period = len(loss_data) // max_iterations
#                     loss_data = loss_data[::period][1:]
#                     losses.append(loss_data)
#         return np.array(losses)
#     else:
#         losses = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if "losses" in file:
#                     loss_file = os.path.join(root, file)
#                     loss_data = np.load(loss_file)
#                     losses.append(loss_data)
#         return np.array(losses)

# # Function to plot average loss curves
# def plot_loss_curves(losses, title, labels):
#     for i, loss in enumerate(losses):
#         mean_loss = np.mean(loss, axis=0)
#         std_loss = np.std(loss, axis=0)
#         plt.plot(mean_loss, label=labels[i])
#         plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
#     plt.title(title)
#     plt.yscale('log')
#     plt.xlabel('Number of evaluations (×5000000)')
#     plt.ylabel('MSE Loss (log scale)')
#     plt.legend()
#     plt.show()

# # Main function to traverse directories and plot loss curves
# def main():
#     functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
#     algorithms = ['GA', 'GA_sharing', 'GA_dynamic']
#     settings = ['param', 'layer', 'node', 'none']
        
#     for func in functions:
#         func_path = os.path.join('results_crossover_ablation', func)
#         func_losses = []
#         labels = []
#         for alg in algorithms:
#             alg_path = os.path.join(func_path, alg)
#             alg_losses = []
#             for setting in settings:
#                 setting_path = os.path.join(alg_path, setting)
#                 alg_losses.append(load_losses(setting_path))
#                 labels.append(f'{alg} ({setting})')
#             func_losses.extend(alg_losses)
#         plt.tight_layout()
#         plot_loss_curves(func_losses, func, labels)

# if __name__ == "__main__":
#     main()


# Plotting the GA_SGD-GA_SGD_sharing-GA_SGD_dynamic plots
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
                    loss_data = loss_data[::period][1:]
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
def plot_loss_curves(losses, title, labels):
    for i, loss in enumerate(losses):
        mean_loss = np.mean(loss, axis=0)
        std_loss = np.std(loss, axis=0)
        plt.plot(mean_loss, label=labels[i])
        plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Number of evaluations (×5000000)')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()

# Main function to traverse directories and plot loss curves
def main():
    functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
    algorithms = ['GA_SGD', 'GA_SGD_sharing', 'GA_SGD_dynamic']
        
    for func in functions:
        func_path = os.path.join('results_GA_SGD_FS', func)
        func_losses = []
        labels = []
        for alg in algorithms:
            alg_path = os.path.join(func_path, alg)
            func_losses.append(load_losses(alg_path))
            labels.append(f'{alg}')
        plt.tight_layout()
        plot_loss_curves(func_losses, func, labels)

if __name__ == "__main__":
    main()
