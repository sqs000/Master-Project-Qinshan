import subprocess
import sys

# Specify the number of repetitions
repetitions = 5

# Specify the BBOB function numbers in a list
# functions = [1, 3, 7, 13, 16, 22]
functions = [3]

# Specify the algorithms
algorithms = ["GA_SGD_sharing"]

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
                    "-p", "200",
                    "-e", "2",
                    "-l", "0.00001",
                    "-b", "64",
                    "-i", str(rep)
                ])
            elif algo == "GA_SGD_sharing":
                # for r in [1, 5, 10, 20, 50]:
                for r in [5]:
                    subprocess.run([
                        sys.executable, "run.py",
                        "--function", str(f),
                        "--algorithm", algo,
                        "--numberofevaluations", "500000000",
                        "-p", "200",
                        "-r", str(r),
                        "-e", "2",
                        "-l", "0.00001",
                        "-b", "64",
                        "-i", str(rep)
                    ])
            elif algo == "GA_SGD_dynamic":
                for n in [5, 10, 20, 50, 100]:
                    subprocess.run([
                        sys.executable, "run.py",
                        "--function", str(f),
                        "--algorithm", algo,
                        "--numberofevaluations", "500000000",
                        "-p", "200",
                        "-r", "5",
                        "-n", str(n),
                        "-e", "2",
                        "-l", "0.00001",
                        "-b", "64",
                        "-i", str(rep)
                    ])
