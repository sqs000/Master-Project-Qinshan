import subprocess
import sys

# Specify the number of repetitions
repetitions = 5

# Specify the BBOB function numbers in a list
functions = [1, 3, 7, 13, 16, 22]

# Specify the algorithms in a list
algorithms = ["GA", "GA_sharing", "GA_dynamic"]

# Repeat the run for the specified number of repetitions
for rep in range(1, repetitions + 1):
    # Loop through the functions
    for f in functions:
        # Loop through the algorithms
        for algo in algorithms:
            if algo == "GA":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-m", "0.04",
                    "-c", "none",
                    "-i", str(rep)
                ])
            elif algo == "GA_sharing":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-m", "0.04",
                    "-c", "none",
                    "-r", "5",
                    "-i", str(rep)
                ])
            elif algo == "GA_dynamic":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-p", "1000",
                    "-m", "0.04",
                    "-c", "none",
                    "-r", "5",
                    "-n", "50",
                    "-i", str(rep)
                ])
