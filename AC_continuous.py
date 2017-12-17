import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.image as img
from car_continuous import Car
import os
import random
import time
try:
    xrange = xrange
except:
    xrange = range

image = img.imread('runtrack5.bmp')[:,:,0]
image = image == 0
track_px = image.astype(int)

r_scale = 0.05

car = Car(track_px,r_scale)

gamma = 0.99
save_freq = 1000
demo_freq = 100
demo_frame_freq = 20
startE = 0.3
endE = 0.05
annealing_steps = 100000

lr = 1e-5
entropy_penalty = 0.0
max_disc_ret = -1e-5
tau = 1e-4
s_size = 8
a_size = 2
h_size = 16

total_episodes = 1e6 #Set total number of episodes to train agent on.
max_ep = 30000
update_frequency = 25
print_freq = 100
total_episodes = 10000 #Set total number of episodes to train agent on.
update_frequency = 1

e = startE
eDrop = (startE - endE)/annealing_steps

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent():
    def __init__(self, lr, s_size,a_size,h_size,entropy_penalty):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        self.action_in = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
        self.v_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

        hidden1 = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        hidden2a = slim.fully_connected(hidden1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        hidden2c = slim.fully_connected(hidden1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #hidden3 = slim.fully_connected(hidden2,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.mu_out = slim.fully_connected(hidden2a,a_size,activation_fn=tf.nn.tanh,biases_initializer=None)
        self.sigma_out = slim.fully_connected(hidden2a,a_size,activation_fn=tf.nn.softplus,biases_initializer=None)
        self.v_out = slim.fully_connected(hidden2c,1,activation_fn=None,biases_initializer=None)
        
        td = tf.subtract(self.v_target,self.v_out)
        self.critic_loss = tf.reduce_mean(tf.square(td))

        normal_dist = tf.contrib.distributions.Normal(self.mu_out,self.sigma_out)

        self.action_out = tf.squeeze(normal_dist.sample(1),axis = 0)
        # calculate action loss:
        log_prob = normal_dist.log_prob(self.action_in)
        self.actor_loss = tf.reduce_mean(-log_prob*td)


        

        actor_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.update_actor = actor_optimizer.minimize(self.actor_loss)
        self.update_critic = critic_optimizer.minimize(self.critic_loss)



tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = Agent(lr=lr,s_size=s_size,a_size=a_size,h_size=h_size,entropy_penalty=entropy_penalty) #Load the agent.



init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Launch the tensorflow graph

with tf.Session() as sess:
    sess.run(init)
    training = False
    i = 0
    j = 0
    total_steps = 0
    total_reward = []
    total_length = []

    # for files in os.listdir("frames"):
    #     fn=os.path.join("frames", files)
    #     os.system("rm -rf "+fn)
    #     break        

    run_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'_'+str(h_size)+'_'+str(lr)+'_'+str(gamma)+'_'+str(max_disc_ret)
     
    while i < total_episodes:
        prev = time.time()
        # output a demo run with still frames to be giffed together later
        if i % demo_freq == 0:
            print('Printing Episode')
            os.makedirs('frames/'+run_name+'/ep_'+str(i).zfill(5)+'/')
            telemetry,r,d = car.reset()
            for j in range(max_ep):

                a = sess.run(myAgent.action_out,feed_dict={myAgent.state_in:[telemetry]})
                
                telemetry,r,d = car.step(a[0])

                if j % demo_frame_freq == 0:
                    car.plotter(j,'Episode '+str(i),'frames/'+run_name+'/ep_'+str(i).zfill(5)+'/'+str(j).zfill(5)+'.png',a[0])

                if d:
                    break
        # end of demo section

        # end of demo section

        telemetry,r,d = car.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            
            
            #     a_dist = sess.run(myAgent.pi_out,feed_dict={myAgent.state_in:[telemetry]})
            # else:
            # a_dist,a = sess.run([myAgent.pi_out,myAgent.chosen_action],feed_dict={myAgent.state_in:[telemetry]})
            a = sess.run(myAgent.action_out,feed_dict={myAgent.state_in:[telemetry]})

            if len(a)==1:
                a = a[0]

            telemetry_new,r,d = car.step(a) #Get our reward for taking an action given a bandit.

            ep_history.append([telemetry,a,r,telemetry_new])
            telemetry = telemetry_new
            running_reward += r
            total_steps += 1
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])

                # ep_history = ep_history[ep_history[:,2]< max_disc_ret,:]

                e -= eDrop*j
                for k in range(j):
                    _,__ = sess.run([myAgent.update_actor,myAgent.update_critic],
                        feed_dict={myAgent.v_target:np.vstack(ep_history[:,2]),
                                myAgent.state_in:np.vstack(ep_history[:,0]),
                                myAgent.action_in:np.vstack(ep_history[:,1])})

                # # add ep_history into big array
                # ep_buffer = np.vstack([ep_buffer,ep_history])

                # if len(ep_buffer) > max_buff_size:
                #     training = True
                #     z = int(len(ep_buffer)-max_buff_size)
                #     ep_buffer = np.delete(ep_buffer,range(z),0)
                    

                # if np.shape(ep_buffer)[0] == max_buff_size:

                # minibatch = ep_buffer[np.random.randint(max_buff_size,size=batch_size),:]
                # feed_dict={myAgent.reward_holder:minibatch[:,2],
                #         myAgent.action_holder:minibatch[:,1],myAgent.state_in:np.vstack(minibatch[:,0])}
                # grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                # for idx,grad in enumerate(grads):
                #     gradBuffer[idx] += grad

                # if i % update_frequency == 0 and i != 0:
                #     feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                #     _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                #     for ix,grad in enumerate(gradBuffer):
                #         gradBuffer[ix] = grad * 0
                    
                total_reward.append(running_reward)
                total_length.append(j)
                break

        
        #Update our running tally of scores.
        if i % 100 == 0:
            print(i,total_steps,np.mean(total_reward[-100:]),np.mean(total_length[-100:]))
        
        if i % save_freq == 0:
            save_path = saver.save(sess,'saved_models/model'+str(i)+'ckpt')


        i += 1





