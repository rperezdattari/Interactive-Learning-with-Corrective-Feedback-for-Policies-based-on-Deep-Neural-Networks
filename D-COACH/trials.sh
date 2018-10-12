#!/bin/bash
c=1
while [ $c -le 30 ]
do
	python Main.py --exp-num $c --error-prob 0.11;
	c=$((c+1))
    echo $c
done
