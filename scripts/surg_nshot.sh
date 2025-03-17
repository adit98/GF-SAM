#!/bin/bash

dataset=${1}
shot=${2}
for fold in 0 1 2;
do
    python main_eval.py  \
        --benchmark ${dataset} \
        --nshot ${shot} \
        --fold ${fold} --log-root "outputs/${dataset}_${shot}shot/fold${fold}" &
done
wait
