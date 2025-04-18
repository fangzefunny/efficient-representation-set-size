#!/bin/bash

## delete the DS.Store
find . -name ".DS_Store" -delete

declare data_sets=("setsize-collins12" "setsize-collins14") 
declare models=(ecRLp) #"RLWM" "ecRL0" "ecRL" "cuRL" "clRL"
declare fit_method='map'
declare alg='BFGS' # Nelder-Mead

for data_set in "${data_sets[@]}"; do
 
    ## step 0: preprocess the data  
    python m0_preprocess.py -d=$data_set

    ## step 2: fit, simulate, and analyze
    for model in "${models[@]}"; do  
        echo Data set=$data_set Model=$model Algorithm=$alg
            python m1_fit.py      -d $data_set -n $model -s 2025 -f 40 -c 40 -m $fit_method -a $alg
            python m2_rollout.py  -d $data_set -n $model -s 2025 -f 10 -c 10 -m $fit_method 
            # if [ "$model" == "ecRL_d" ]; then
            #     python m3_recover.py  -d $data_set -n $model -s 2025 -c 10 -t "param" -m $fit_method -a $alg
            # fi
    done
done
