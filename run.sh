#!/bin/bash

# Specify the number of repetitions
repetitions=1

# Specify the BBOB function numbers in an array
functions=(1 3 7 13 16 22)
# Specify the algorithms
algorithms=(Adam SGD GA GA_sharing GA_dynamic)

# Loop through the functions
for f in "${functions[@]}"
do
    # Loop through the algorithms
    for algo in "${algorithms[@]}"
    do
        # Repeat the run for the specified number of repetitions
        for ((rep=1; rep<=$repetitions; rep++))
        do
	    fname=${algo}_F${f}_${rep}
            if [ "$algo" == "Adam" ]; then
                python run.py --function $f --algorithm $algo --numberofevaluations 500000000 -l 0.00001 -b 64 -i $rep > ${fname}.out 2> ${fname}.err
            elif [ "$algo" == "SGD" ]; then
                python run.py --function $f --algorithm $algo --numberofevaluations 500000000 -l 0.00001 -b 64 -i $rep > ${fname}.out 2> ${fname}.err
            elif [ "$algo" == "GA" ]; then
                python run.py --function $f --algorithm $algo --numberofevaluations 500000000 -p 1000 -m 0.04 -i $rep > ${fname}.out 2> ${fname}.err
            elif [ "$algo" == "GA_sharing" ]; then
                python run.py --function $f --algorithm $algo --numberofevaluations 500000000 -p 1000 -m 0.04 -r 5 -i $rep > ${fname}.out 2> ${fname}.err
            elif [ "$algo" == "GA_dynamic" ]; then
                python run.py --function $f --algorithm $algo --numberofevaluations 500000000 -p 1000 -m 0.04 -r 5 -n 50 -i $rep > ${fname}.out 2> ${fname}.err
            fi
        done
    done    
done
ps aux | grep run.py | grep -v grep | wc -l

