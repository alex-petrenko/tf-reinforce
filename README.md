# tf-reinforce

Tensorflow implementation of the REINFORCE policy gradient algorithm,
developed as a class project for the course "Stochastic Processes and Reinforcement Learning".
See the [PDF project report](https://github.com/alex-petrenko/tf-reinforce/blob/master/files/project_report.pdf)
for details. 

## How-to:

#### MountainCar-v0

Train a REINFORCE agent on Gym MountainCar-v0 environment (feel free to stop training when the desired episode
reward is achieved, it tops at around -105 to -110 on average, but it may take a long time):

```shell
python -m train_mountaincar
```

Monitor the training progress in Tensorboard:

```shell
tensorboard --logdir .experiments/
```

Model with the best average reward will be saved to the filesystem, use the following command to visualize
the policy behavior:

```shell
python -m enjoy_mountaincar
```

![mountaincar](https://github.com/alex-petrenko/tf-reinforce/blob/master/files/car.gif?raw=true)


#### CartPole-v0

Train on Gym CartPole-v0 environment
(trains much faster than MountainCar, mere 200 episodes should be enough for a good policy):

```shell
python -m train_cartpole
```

Monitor the training progress:

```shell
tensorboard --logdir .experiments/
```

Use the following command to visualize the behavior:

```shell
python -m enjoy_cartpole
```

![mountaincar](https://github.com/alex-petrenko/tf-reinforce/blob/master/files/pole.gif?raw=true)



If you have any questions please feel free to reach me: apetrenko1991@gmail.com
