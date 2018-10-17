# Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks
Code of the paper "Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks" to be presented at ISER 2018.

This code is based on the following publication:
1. [Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks (Preprint)](https://arxiv.org/abs/1810.00466) 

**Authors:** Rodrigo Pérez-Dattari, Carlos Celemin, Javier Ruiz-del-Solar, Jens Kober.

Link to paper video:

[![Paper Video](https://img.youtube.com/vi/vcEtuRrRIe4/0.jpg)](https://www.youtube.com/watch?v=vcEtuRrRIe4)

## Installation

To use the code, it is necessary to first install the gym toolkit (release v0.9.6): https://github.com/openai/gym

Then, the files in the `gym` folder of this repository should be replaced/added in the installed gym folder on your PC. There are modifications of two gym environments:

1. **Continuous-CartPole:** a continuous-action version of the Gym CartPole environment.

2. **CarRacing:** the same CarRacing environment of Gym with some bug fixes and modifications in the main loop for database generation.

To download and install some pretrained networks (CarRacing autoencoder, CarRacing simulated teacher, CartPole simulated teacher) and a CarRacing database for training the autoencoder run (inside the folder `D-COACH`):

```bash 
sh install_comp_files.sh
```

### Requirements
* setuptools==38.5.1
* numpy==1.13.3
* opencv_python==3.4.0.12
* matplotlib==2.2.2
* tensorflow==1.4.0
* pyglet==1.3.2
* gym==0.9.6

## Usage

1. To run the main program type in the terminal (inside the folder `D-COACH`):

```bash 
python main.py --config-file <environment>
```
The default configuration files are **car_racing** and **cartpole**.

To be able to give feedback to the agent, the environment rendering window must be selected/clicked.

To train the autoencoder for the high-dimensional state environments run (inside the folder `D-COACH`):

```bash 
python autoencoder.py
```
2. To generate a database for the CarRacing environment run the (replaced) file `car_racing.py` in the downloaded gym repository.

To modify the dimension of the images in the generated database, this database must be in the folder `D-COACH` and from this folder run:

```bash 
python tools/transform_database_dim.py
```

## Comments

The D-COACH algorithm is designed to work with continuous-action problems. Given that the Cartpole environment of gym was designed to work with discrete action spaces, a modified continuous version of this environment is used.

This code has been tested in `Ubuntu 16.04` and `python >= 3.5`.

Tests with the CartPole environment using the classic COACH algorithm can be done using the following repository: [COACH](https://github.com/rperezdattari/COACH-gym)

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/rperezdattari/Interactive-Learning-with-Corrective-Feedback-for-Policies-based-on-Deep-Neural-Networks/issues) on this repository. It is quite possible that you have run into some bug we are not aware of.

