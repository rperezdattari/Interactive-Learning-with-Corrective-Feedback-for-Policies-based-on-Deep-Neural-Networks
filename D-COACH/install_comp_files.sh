#!/bin/bash

mkdir -p graphs/teacher/CartPole-v0
mkdir -p graphs/teacher/CarRacing-v0
mkdir -p graphs/autoencoder/CarRacing-v0

https://drive.google.com/file/d/1US_dzwgVXVxY4LosyKj1E9u1VGU_phkY/view?usp=sharing

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=133VZfn9U9pHVZXknBJ6I5qwZvzQ9Yh8g' -O graphs/autoencoder/CarRacing-v0/autoencoder.zip

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bNhwyBdgl6OAdJ6HEpaFLYOVzV4SX-hn' -O graphs/teacher/CarRacing-v0/teacher.zip

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1US_dzwgVXVxY4LosyKj1E9u1VGU_phkY' -O graphs/teacher/CartPole-v0/teacher.zip

unzip -d graphs/autoencoder/CarRacing-v0 graphs/autoencoder/CarRacing-v0/autoencoder.zip 
unzip -d graphs/teacher/CarRacing-v0 graphs/teacher/CarRacing-v0/teacher.zip 
unzip -d graphs/teacher/CartPole-v0 graphs/teacher/CartPole-v0/teacher.zip 

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kYOVI1CiU4fzjEJjkZN-R4P92hP3y2HL' -O racing_car_classic_database_64x64.npy
