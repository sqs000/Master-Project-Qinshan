# Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for all functions
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


# Plotting the SGD-Adam-GA-GA_sharing-GA_dynamic(param/node/layer) plots for F3
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


# Plotting the GA-GA_sharing-GA_dynamic(param/node/layer/none) plots
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
        plot_loss_curves(func_losses, func, labels)

if __name__ == "__main__":
    main()



# # Plotting the SGD,Adam,GA,GA_sharing,GA_dynamic(param/node/layer) plots
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load loss data from npy files
# def load_losses(folder_path):
#     losses = []
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if "losses" in file:
#                 loss_file = os.path.join(root, file)
#                 loss_data = np.load(loss_file)
#                 losses.append(loss_data)
#     return np.array(losses)

# # Function to plot average loss curves
# def plot_loss_curves(losses, title):
#     avg_losses = np.mean(losses, axis=0)
#     plt.plot(avg_losses.T)
#     plt.title(title)
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.show()

# # Main function to traverse directories and plot loss curves
# def main():
#     functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
#     algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
#     settings = ['param', 'layer', 'node']
    
#     for func in functions:
#         fig, axs = plt.subplots(figsize=(10, 6))
#         func_path = os.path.join('results3', func)
#         for alg in algorithms:
#             alg_path = os.path.join(func_path, alg)
#             if alg == 'SGD' or alg == 'Adam':
#                 losses = load_losses(alg_path)
#                 plot_loss_curves(losses, f'{func} - {alg}')
#             else:
#                 for setting in settings:
#                     setting_path = os.path.join(alg_path, setting)
#                     losses = load_losses(setting_path)
#                     plot_loss_curves(losses, f'{func} - {alg} ({setting})')

# if __name__ == "__main__":
#     main()

# # Plotting the SGD-Adam / GA-GA_sharing-GA_dynamic(param/node/layer) plots
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load loss data from npy files
# def load_losses(folder_path, max_iterations=None):
#     losses = []
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if "losses" in file:
#                 loss_file = os.path.join(root, file)
#                 loss_data = np.load(loss_file)[:max_iterations]
#                 losses.append(loss_data)
#     return np.array(losses)

# # Function to plot average loss curves
# def plot_loss_curves(losses, title, labels):
#     for i, loss in enumerate(losses):
#         plt.plot(np.mean(loss, axis=0), label=labels[i])
#     plt.title(title)
#     plt.yscale('log')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# # Main function to traverse directories and plot loss curves
# def main():
#     functions = ['F1', 'F3', 'F7', 'F13', 'F16', 'F22']
#     algorithms = ['Adam', 'SGD', 'GA', 'GA_sharing', 'GA_dynamic']
#     settings = ['param', 'layer', 'node']
        
#     for func in functions:
#         ga_losses = []
#         sgd_adam_losses = []
#         ga_labels = []
#         sgd_adam_labels = []
        
#         func_path = os.path.join('results3', func)
        
#         # Load SGD and Adam losses
#         for alg in algorithms[:2]:
#             alg_path = os.path.join(func_path, alg)
#             sgd_adam_losses.append(load_losses(alg_path))
#             sgd_adam_labels.append(alg)
        
#         # Load GA losses
#         for alg in algorithms[2:]:
#             for setting in settings:
#                 alg_path = os.path.join(func_path, alg, setting)
#                 ga_losses.append(load_losses(alg_path))
#                 ga_labels.append(f'{alg} ({setting})')
                
#         # Plot SGD and Adam
#         plot_loss_curves(sgd_adam_losses, f'{func} - SGD & Adam', sgd_adam_labels)
        
#         # Plot GAs
#         plot_loss_curves(ga_losses, f'{func} - Genetic Algorithms', ga_labels)
        
# if __name__ == "__main__":
#     main()

