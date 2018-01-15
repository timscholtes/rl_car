## Synopsis

This repo contains code to instantiate a custom sandbox environment for driving a car (a dot with an arrow) with either continuous (car_continuous) or discrete (car_LCR) control around a custom track that can be made in paint (eg runtrack5.bmp)

Several examples of new RL algorithms are implemented in Tensorflow to learn effective policies to drive around the track. The most successful has been A3C (async_continuous.py), running simply on my MacBook, which is able to learn to avoid crashing and decelerate for corners, completing full laps of the track, after just a few hours of training

## A3C Setup

To encourage good behaviour, I set the reward function to be -1 for a crash, and alpha * speed / max_speed at all other times, where alpha is a tuning parameter. Note I didn't have to define speed in the direction of the track, as the narrowness of the track ensures the global optimum is to follow the road rather than small circles on the start line. Speed is allowed to be negative and incurs negative reward.

I used gamma = 0.99 for standard method of discounting the reward back through time. This function is interesting as it, along with the relative scale of the rewards for crashing and speeding, heavily determine the training effectiveness. I'm interested to see in future work if a more sophisticated reward function can be learned. For instance, the notion that credit assignment follows exponentially decaying behaviour rather than having sharp drop-offs is quite a thing - although to my mind the inclusion of a value baseline in calculating the advantage, if well learned, goes a great way to helping this.

## Results

The learning is not the most stable - while it is able to learn optimal policies, it can just as easily forget them. I'm sure with more time to find suitable hyperparameters, and a beefier machine, this could be easily overcome. I'd be interested to explore whether simply adding more parallel actors than the 4 my MacBook can support would greatly improve stability. See below for the reward values as a function of time (split into two as I suffered a training interruption).

See also a gif of final performance on the track. The track positioning and speed vector are in the top left, and a plot of historical speed against distance covered is in the bottom left. Top right is the policy outpout (as a gaussian distribution in acceleration and steering), with bottom left being the input to the policy and value networks (actor/critic), which is the distance of angled rays outwards from the car until a wall, as well as current speed.

![](https://github.com/timscholtes/rl_car/blob/master/movie.gif)

## Credit

To learn these techniques, I followed the **excellent** tutorial by Arthur Juliani [here](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0).

Some of the code is heavily adapted from his examples.
