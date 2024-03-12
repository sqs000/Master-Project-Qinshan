#!/bin/bash

# Specify the number of repetitions
repetitions=10

# Specify the BBOB function numbers in an array
functions=(3 7 13 16 22)

# Loop through the functions
for f in "${functions[@]}"
do
    # Repeat the run for the specified number of repetitions
    for ((rep=1; rep<=$repetitions; rep++))
    do
        python run.py $f GA 1000000 -p 1000 -m 0.04 -i $rep
    done
    
done
