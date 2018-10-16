#!/bin/bash
c=1
while [ $c -le 30 ]
do
	python main.py --config-file cartpole;
	c=$((c+1))
    echo $c
done
