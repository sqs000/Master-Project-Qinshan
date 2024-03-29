#!/bin/bash

# Specify the number of repetitions
repetitions=5

# Specify the BBOB function numbers in an array
functions=(1 3 7 13 16 22)
# Specify the algorithms
algorithms=(GA)

# Specify population sizes, mutation rates, and crossover types to experiment with
population_sizes=(500 1000 2000)
mutation_rates=(0.02 0.04 0.06)
crossover_types=("param" "node" "layer")

# Repeat the run for the specified number of repetitions
for ((rep=1; rep<=$repetitions; rep++))
do
    # Loop through the functions
    for f in "${functions[@]}"
    do
        # Loop through the algorithms
        for algo in "${algorithms[@]}"
        do
            # Loop through population sizes
            for pop_size in "${population_sizes[@]}"
            do
                # Loop through mutation rates
                for mutation_rate in "${mutation_rates[@]}"
                do
                    # Loop through crossover types
                    for crossover_type in "${crossover_types[@]}"
                    do
                        fname=${algo}_F${f}_${rep}_P${pop_size}_M${mutation_rate}_C${crossover_type}
                        python run.py --function $f --algorithm $algo --numberofevaluations 2500000000 -p $pop_size -m $mutation_rate -c $crossover_type -i $rep > ${fname}.out 2> ${fname}.err
                    done
                done
            done
        done
    done    
done
ps aux | grep run.py | grep -v grep | wc -l
