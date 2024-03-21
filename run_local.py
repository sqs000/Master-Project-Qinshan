import subprocess
import sys

# Specify the number of repetitions
repetitions = 10

# Specify the BBOB function numbers in a list
functions = [1, 3, 7, 13, 16, 22]

# Specify the algorithms in a list
algorithms = ["Adam", "SGD", "GA", "GA_sharing", "GA_dynamic"]

# Loop through the functions
for f in functions:
    # Loop through the algorithms
    for algo in algorithms:
        # Repeat the run for the specified number of repetitions
        for rep in range(1, repetitions + 1):
            if algo == "Adam" or algo == "SGD":
                subprocess.run([
                    sys.executable, "run.py",
                    "--function", str(f),
                    "--algorithm", algo,
                    "--numberofevaluations", "500000000",
                    "-l", "0.00001",
                    "-b", "64",
                    "-i", str(rep)
                ])
            elif algo == "GA":
                for c in ["param", "layer", "node"]:
                    subprocess.run([
                        sys.executable, "run.py",
                        "--function", str(f),
                        "--algorithm", algo,
                        "--numberofevaluations", "500000000",
                        "-p", "1000",
                        "-m", "0.04",
                        "-c", c,
                        "-i", str(rep)
                    ])
            elif algo == "GA_sharing":
                for c in ["param", "layer", "node"]:
                    subprocess.run([
                        sys.executable, "run.py",
                        "--function", str(f),
                        "--algorithm", algo,
                        "--numberofevaluations", "500000000",
                        "-p", "1000",
                        "-m", "0.04",
                        "-c", c,
                        "-r", "5",
                        "-i", str(rep)
                    ])
            elif algo == "GA_dynamic":
                for c in ["param", "layer", "node"]:
                    subprocess.run([
                        sys.executable, "run.py",
                        "--function", str(f),
                        "--algorithm", algo,
                        "--numberofevaluations", "500000000",
                        "-p", "1000",
                        "-m", "0.04",
                        "-c", c,
                        "-r", "5",
                        "-n", "50",
                        "-i", str(rep)
                    ])
