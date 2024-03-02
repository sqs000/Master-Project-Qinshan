import numpy as np

# sgd_final_losses = np.load("sgd_results.npy")
ga_final_losses = np.load("results/GA/ga_results.npy")
ga_sharing_final_losses = np.load("results/GA/ga_sharing_results.npy")
ga_dynamic_final_losses = np.load("results/GA/ga_dynamic_results.npy")

mean_loss = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
mean_loss["GA"] = np.mean(ga_final_losses)
mean_loss["GA_sharing"] = np.mean(ga_sharing_final_losses)
mean_loss["GA_dynamic"] = np.mean(ga_dynamic_final_losses)

best_loss = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
best_loss["GA"] = np.min(ga_final_losses)
best_loss["GA_sharing"] = np.min(ga_sharing_final_losses)
best_loss["GA_dynamic"] = np.min(ga_dynamic_final_losses)

std_deviation = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
std_deviation["GA"] = np.std(ga_final_losses, ddof=1)
std_deviation["GA_sharing"] = np.std(ga_sharing_final_losses, ddof=1)
std_deviation["GA_dynamic"] = np.std(ga_dynamic_final_losses, ddof=1)

std_error = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
std_error["GA"] = np.std(ga_final_losses, ddof=1) / np.sqrt(len(ga_final_losses))
std_error["GA_sharing"] = np.std(ga_sharing_final_losses, ddof=1) / np.sqrt(len(ga_sharing_final_losses))
std_error["GA_dynamic"] = np.std(ga_dynamic_final_losses, ddof=1) / np.sqrt(len(ga_dynamic_final_losses))

oneQ = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
oneQ["GA"] = np.percentile(ga_final_losses, 25)
oneQ["GA_sharing"] = np.percentile(ga_sharing_final_losses, 25)
oneQ["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 25)

median = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
median["GA"] = np.percentile(ga_final_losses, 50)
median["GA_sharing"] = np.percentile(ga_sharing_final_losses, 50)
median["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 50)

threeQ = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
threeQ["GA"] = np.percentile(ga_final_losses, 75)
threeQ["GA_sharing"] = np.percentile(ga_sharing_final_losses, 75)
threeQ["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 75)

# List of dictionaries
list_of_dicts = [mean_loss, best_loss, std_deviation, std_error, oneQ, median, threeQ]

# Iterate through the list and print each dictionary
for idx, d in enumerate(list_of_dicts, start=1):
    print(f"Dictionary {idx}:")
    for key, value in d.items():
        print(f"  {key}: {value}")
    print()
