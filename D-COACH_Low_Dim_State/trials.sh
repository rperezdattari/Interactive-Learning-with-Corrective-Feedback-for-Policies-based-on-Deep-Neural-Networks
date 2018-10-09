#!/bin/bash
c=1
while [ $c -le 30 ]
do
	python Main.py $c /NN_teacher_F_buffer NN True;
	c=$((c+1))
done

c=1
while [ $c -le 30 ]
do
	python Main.py $c /rbf_teacher_F linear_RBFs False;
	c=$((c+1))
done
