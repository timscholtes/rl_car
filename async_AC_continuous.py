import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.image as img
import multiprocessing
import threading
from car_continuous import Car
import os
import random
import time
try:
    xrange = xrange
except:
    xrange = range

# load in the track
image = img.imread('tracks/runtrack5.bmp')[:,:,0]
image = image == 0
track_px = image.astype(int)

# set the parameters for the learning algo, and the saving scheme
r_scale = 0.01
gamma = 0.99
save_freq = 1000
demo_freq = 100
demo_frame_freq = 10

lr = 1e-3

max_disc_ret = -1e-5
tau = 1e-4
s_size = 8
a_size = 2
h_size = 16

frame_repeat = 4

max_ep = 3000
max_dist = 5000
update_frequency = 25
print_freq = 100
total_episodes = 10000 #Set total number of episodes to train agent on.
update_frequency = 1

annealing_steps = 1000
entropy_penalty_start = 0# 0.01
entropy_penalty_end = 0
eDrop = (entropy_penalty_start - entropy_penalty_end)/annealing_steps

# This func takes globally held graph and sends it to the local threads
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




###### - defines the network, as well as the loss function (for the local threads to use when updating the network)
class Agent():
    def __init__(self, s_size,a_size,h_size,actor_trainer,critic_trainer,scope):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        with tf.variable_scope(scope):
            self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.action_in = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
            self.v_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

            hidden1 = slim.fully_connected(self.state_in,h_size,activation_fn=tf.nn.relu,biases_initializer=None)
            hidden2a = slim.fully_connected(hidden1,h_size,activation_fn=tf.nn.relu,biases_initializer=None)
            hidden2c = slim.fully_connected(hidden1,h_size,activation_fn=tf.nn.relu,biases_initializer=None)
            #hidden3 = slim.fully_connected(hidden2,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
            self.mu_out = slim.fully_connected(hidden2a,a_size,activation_fn=tf.nn.tanh,biases_initializer=None)
            self.sigma_out = slim.fully_connected(hidden2a,a_size,activation_fn=tf.nn.softplus,biases_initializer=None)
            self.v_out = slim.fully_connected(hidden2c,1,activation_fn=None,biases_initializer=None)

            if scope != 'global':

                td = tf.subtract(self.v_target,self.v_out)
                self.critic_loss = tf.reduce_mean(tf.square(td))

                normal_dist = tf.contrib.distributions.Normal(self.mu_out,self.sigma_out)

                self.action_out = tf.squeeze(normal_dist.sample(1),axis = 0)
                # calculate action loss:
                log_prob = normal_dist.log_prob(self.action_in)
                entropy = normal_dist.entropy()
                self.entropy_penalty_scaled = tf.placeholder(shape=[None],dtype=tf.float32)
                self.actor_loss = tf.reduce_mean(-log_prob*td) - self.entropy_penalty_scaled * entropy

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
                actor_gradients = tf.gradients(self.actor_loss,local_vars)
                critic_gradients = tf.gradients(self.critic_loss,local_vars)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'global')
                self.apply_actor_grads = actor_trainer.apply_gradients(zip(actor_gradients,global_vars))
                self.apply_critic_grads = critic_trainer.apply_gradients(zip(critic_gradients,global_vars))

# each worker inherits the global network, as well as it's own car.
# The local worker drives on the course, and then updates the global network with the grads from its own local network.
class Worker():
    def __init__(self,car,name,actor_trainer,critic_trainer,global_episodes,entropy_penalty_start,eDrop):
        self.car = car
        self.name = "worker" + str(name)
        self.number = name
        self.entropy_penalty = entropy_penalty_start
        self.local_agent = Agent(
            s_size=s_size,
            a_size=a_size,
            h_size=h_size,
            actor_trainer=actor_trainer,
            critic_trainer=critic_trainer,
            scope=self.name)
        self.update_local_ops = update_target_graph('global',self.name)
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1) #to be run by worker 0
        self.summary_writer = tf.summary.FileWriter('async_TB_logs/train_'+str(self.number))


    def demo_save(self,sess,run_name,episode_count):
        print('Printing Episode',str(episode_count))
        os.makedirs('frames/'+run_name+'/ep_'+str(episode_count).zfill(5)+'/')
        telemetry,r,d = self.car.reset()
        accumulator = 0
        for j in range(max_ep):
            a,sigma = sess.run([self.local_agent.mu_out,self.local_agent.sigma_out],feed_dict={self.local_agent.state_in:[telemetry]})
            telemetry,r,d = self.car.step(a[0],recording=True)
            extra = [a[0][0],a[0,1],sigma[0][0],sigma[0][1]]
            if j % demo_frame_freq == 0:
                print('Saving episode ',episode_count,', frame ',j)
                self.car.plotter2(j,'Episode '+str(episode_count),
                    'frames/'+run_name+'/ep_'+str(episode_count).zfill(5)+'/'+str(j).zfill(5)+'.png',extra)

            # Check for stalled cars!
            if self.car.s == 0:
                accumulator += 1
            else:
                accumulator = 0

            # End the loop if done or stalled
            if accumulator == 3:
                self.car.prev_speed_dist_histories.append([self.car.speed_history,self.car.dist_history])
                d = True
            if d:
                print('done')
                break

    def model_save(self,sess,run_name,episode_count):
        saver.save(sess,'./models/'+run_name+'/model',global_step=episode_count)

    def drive(self,sess,coord,run_name):

        episode_count = sess.run(self.global_episodes)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)

                # Write out frames to demo the agent at this point.
                if episode_count % demo_freq == 0 and self.name == 'worker0':
                    self.demo_save(sess,run_name,episode_count)
                    self.model_save(sess,run_name,episode_count)
                    sess.run(self.increment)
                    episode_count = sess.run(self.global_episodes)

                # Now run the training - drive on the car and update the network.
                telemetry,r,d = self.car.reset()
                ep_history = []
                episode_reward = 0
                accumulator = 0
                avg_speed = self.car.s
                j = 0
                for j in range(max_ep): #while j < max_ep:
                    #Probabilistically pick an action given our network outputs.
                    a = sess.run(self.local_agent.action_out,feed_dict={self.local_agent.state_in:[telemetry]})
                    if len(a)==1:
                        a = a[0]

                    # for k in range(frame_repeat):
                    telemetry_new,r,d = self.car.step(a) #Get our reward for taking an action given a bandit.

                    # Check for stalled cars - if it has stalled, give it the same negative reward as a crash
                    if self.car.s == 0:
                        accumulator += 1
                    else:
                        accumulator = 0
                    if accumulator == 10:
                        d = True
                        r = -0.5

                    ep_history.append([telemetry,a,r,telemetry_new])
                    telemetry = telemetry_new
                    episode_reward += r
                    avg_speed += self.car.s
                    # j += 1
                    # if d:
                    #     break

                    # Now the training run is over, update the network
                    if d == True:
                        self.entropy_penalty -= eDrop
                        episode_count = sess.run(self.global_episodes)
                        ep_history = np.array(ep_history)
                        ep_history[:,2] = discount_rewards(ep_history[:,2])

                        _,__ = sess.run([self.local_agent.apply_actor_grads,
                            self.local_agent.apply_critic_grads],
                            feed_dict={self.local_agent.v_target: np.vstack(ep_history[:,2]),
                                    self.local_agent.state_in: np.vstack(ep_history[:,0]),
                                    self.local_agent.action_in: np.vstack(ep_history[:,1]),
                                    self.local_agent.entropy_penalty_scaled: [self.entropy_penalty]})
                        
                        # Save some summary statistics
                        summary = tf.Summary()
                        summary.value.add(tag='Reward',simple_value = float(episode_reward))
                        summary.value.add(tag='Length',simple_value = float(j))
                        summary.value.add(tag='Distance',simple_value = float(self.car.d))
                        summary.value.add(tag='Avg Speed',simple_value = float(avg_speed/j))
                        self.summary_writer.add_summary(summary,episode_count)
                        self.summary_writer.flush()

                        # increase the global episode counter (once for the whole set of workers)
                        # This is done on worker1 as sometimes worker0 is saving down a demo
                        if self.name == 'worker0':
                            sess.run(self.increment)

                        break

#### RUN THE PROGRAM!
tf.reset_default_graph() #Clear the Tensorflow graph.

run_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'_'+str(h_size)+'_'+str(lr)+'_'+str(gamma)+'_'+str(max_disc_ret)

# if not os.path.exists('./frames/'+str(run_name)):
#     os.makedirs('./frames/'+str(run_name))
if not os.path.exists('./models/'+str(run_name)):
    os.makedirs('./models/'+str(run_name))

# set up the trainers, the environments and the workers
with tf.device('/cpu:0'):

    #lr, s_size,a_size,h_size,entropy_penalty,actor_trainer,critic_trainer,scope
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    actor_trainer = tf.train.AdamOptimizer(lr)
    critic_trainer = tf.train.AdamOptimizer(lr)
    master_agent = Agent(s_size=s_size,
    a_size=a_size,
    h_size=h_size,
    actor_trainer=actor_trainer,
    critic_trainer=critic_trainer,
    scope='global') #Load the agent.

    num_workers = multiprocessing.cpu_count()
    workers = []

    for i in range(num_workers):
        workers.append(Worker(Car(track_px,r_scale),i,actor_trainer,critic_trainer,global_episodes,entropy_penalty_start,eDrop))
    saver = tf.train.Saver()

# start the threads
init = tf.global_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(init)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.drive(sess,coord,run_name)
        t = threading.Thread(target=(worker_work))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)






