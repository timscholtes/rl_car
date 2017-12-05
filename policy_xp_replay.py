import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from car import Car
import os



try:
    xrange = xrange
except:
    xrange = range

image = img.imread('run_track2.bmp')[:,:,0]
image = image == 0
track_px = image.astype(int)

car = Car(track_px)

gamma = 0.99
save_freq = 500
demo_freq = 250
demo_frame_freq = 10
e = 0.1

lr = 1e-2
s_size = 8
a_size = 9
h_size = 32
batch_size = 1e3
max_buff_size = 1e5

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden1 = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #hidden2 = slim.fully_connected(hidden1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #hidden3 = slim.fully_connected(hidden2,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden1,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))



tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=lr,s_size=s_size,a_size=a_size,h_size=h_size) #Load the agent.

total_episodes = 10000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 1

init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Launch the tensorflow graph
ep_buffer = np.array([]).reshape(0,4)
with tf.Session() as sess:
    sess.run(init)
    training = False
    i = 0
    total_reward = []
    total_length = []

    for files in os.listdir("frames"):
        fn=os.path.join("frames", files)
        os.system("rm -rf "+fn)
        break
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:

        # output a demo run with still frames to be giffed together later
        if i % demo_freq == 0 and training:
            print(i,training)
            os.makedirs('frames/ep_'+str(i)+'/')
            telemetry,r,d = car.reset()

            for j in range(max_ep):
                if j % demo_frame_freq == 0:
                    car.plotter(j,'Episode '+str(i),'frames/ep_'+str(i)+'/'+str(j).zfill(5)+'.png')
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[telemetry]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)
                
                telemetry_new,r,d = car.step(a)
                if d:
                    break
        # end of demo section

        telemetry,r,d = car.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            
            if np.random.rand(1) < e:
                a = np.random.randint(9)
            else:
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[telemetry]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)


            telemetry_new,r,d = car.step(a) #Get our reward for taking an action given a bandit.

            ep_history.append([telemetry,a,r,telemetry_new])
            telemetry = telemetry_new
            running_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])

                # add ep_history into big array
                ep_buffer = np.vstack([ep_buffer,ep_history])
                if len(ep_buffer) > max_buff_size:
                    training = True
                    z = int(len(ep_buffer)-max_buff_size)
                    ep_buffer = np.delete(ep_buffer,range(z),0)
                    

                if np.shape(ep_buffer)[0] == max_buff_size:
                    minibatch = ep_buffer[np.random.randint(max_buff_size,size=batch_size),:]
                    feed_dict={myAgent.reward_holder:minibatch[:,2],
                            myAgent.action_holder:minibatch[:,1],myAgent.state_in:np.vstack(minibatch[:,0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                    
                total_reward.append(running_reward)
                total_length.append(j)
                break

        
        #Update our running tally of scores.
        if i % 100 == 0 and training:
            print(i,np.mean(total_reward[-100:]))
        if not training and i % 100 == 0:
            print(i,len(ep_buffer),'Not yet training')
        if i % save_freq == 0:
            save_path = saver.save(sess,'saved_models/model'+str(i)+'ckpt')


        i += 1





