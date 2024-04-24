import subprocess
import sys

# Specify the number of repetitions
repetitions = 5

# Specify the BBOB function numbers in a list
functions = [1, 3, 7, 13, 16, 22]

# Specify the algorithm and hyperparameters
algorithms = ["GA_SGD", "GA_SGD_sharing", "GA_SGD_dynamic"]

# Repeat the run for the specified number of repetitions
for rep in range(1, repetitions + 1):
    # Loop through the functions
    for f in functions:
        # Loop through the algorithms
        for algo in algorithms:
            if algo == "GA_SGD":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-e", "1",
                    "-l", "0.00001",
                    "-b", "64",
                    "-i", str(rep)
                ])
            elif algo == "GA_SGD_sharing":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-r", "5",
                    "-e", "1",
                    "-l", "0.00001",
                    "-b", "64",
                    "-i", str(rep)
                ])
            elif algo == "GA_SGD_dynamic":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-r", "5",
                    "-n", "50",
                    "-e", "1",
                    "-l", "0.00001",
                    "-b", "64",
                    "-i", str(rep)
                ])
