import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from car_LCR import Car
import os
import random
import time
try:
    xrange = xrange
except:
    xrange = range

image = img.imread('run_track3.bmp')[:,:,0]
image = image == 0
track_px = image.astype(int)

car = Car(track_px)

gamma = 0.99
pre_train_length = 1e3
save_freq = 1000
demo_freq = 1000000
demo_frame_freq = 10
startE = 1
endE = 0.1
annealing_steps = 10000

lr = 1e-2
tau = 0.001
batch_size = 64
buffer_size = 50000
s_size = 8
a_size = 3
h_size = 32

total_episodes = 1e6 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 25
print_freq = 100

e = startE
eDrop = (startE - endE)/annealing_steps


class Agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden1 = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        hidden2 = slim.fully_connected(hidden1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        # hidden3 = slim.fully_connected(hidden2,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.Q_out = slim.fully_connected(hidden2,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.predict = tf.argmax(self.Q_out,1)



        self.targetQ = tf.placeholder(shape=[None],dtype = tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype = tf.float32)

        self.actions_oh = tf.one_hot(self.actions,a_size,dtype=tf.int32)
        self.Q_eval = tf.reduce_sum(tf.multiply(self.Q_out,self.actions_oh), axis = 1)

        self.loss = tf.reduce_sum(tf.square(self.targetQ-self.Q_eval))
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.update_batch = optimizer.minimize(self.loss)

class Experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,size):
        s = random.sample(self.buffer,size)
        return np.reshape(np.array(s),[size,5])

class Experience_buffer_set_list():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[random.randrange(self.buffer_size)] = experience
            
    def sample(self,size):
        s = random.sample(self.buffer,size)
        return np.reshape(np.array(s),[size,5])


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

tf.reset_default_graph() #Clear the Tensorflow graph.

#myAgent = agent(lr=1e-2,s_size=8,a_size=9,h_size=64) #Load the agent.
actor = Agent(lr=lr,s_size=s_size,a_size=a_size,h_size=h_size) #Load the agent.
critic = Agent(lr=lr,s_size=s_size,a_size=a_size,h_size=h_size) #Load the agent.
xp_buffer = Experience_buffer(buffer_size)
# xp_buffer = Experience_buffer_set_list(buffer_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_steps = 0
    total_reward = []
    total_length = []

    # for files in os.listdir("frames"):
    #     fn=os.path.join("frames", files)
    #     os.system("rm -rf "+fn)
    #     break
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        prev = time.time()
        # output a demo run with still frames to be giffed together later
        # if i % demo_freq == 0:
        #     os.makedirs('frames/ep_'+str(i)+'/')
        #     telemetry,r,d = car.reset()

        #     for j in range(max_ep):
        #         if j % demo_frame_freq == 0:
        #             car.plotter(j,'Episode '+str(i),'frames/ep_'+str(i)+'/'+str(j).zfill(5)+'.png')


        #         a_dist = sess.run(myAgent.Q_out,feed_dict={myAgent.state_in:[telemetry]})

        #         a = np.random.choice(a_dist[0],p=a_dist[0])
        #         a = np.argmax(a_dist == a)
                
        #         telemetry_new,r,d = car.step(a)
        #         if d:
        #             break
        # end of demo section

        telemetry,r,d = car.reset()
        running_reward = 0
        
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            
            
            a_dist = sess.run(actor.Q_out,feed_dict={actor.state_in:[telemetry]})
            a = 1
            if np.random.rand(1) < e:
                a = np.random.randint(a_size)
            else:
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)

            telemetry_new,r,d = car.step(a) #Get our reward for taking an action given a bandit.

            xp_buffer.add([telemetry,r,d,int(a),telemetry_new])
            
            telemetry = telemetry_new
            running_reward += r
            total_steps += 1

            if total_steps > pre_train_length:
                
                if e > endE:
                    e -= eDrop

                if total_steps % update_frequency == 0:

                    train_batch = xp_buffer.sample(batch_size)                

                    chosen_actions = sess.run(actor.predict,feed_dict={actor.state_in: np.vstack(train_batch[:,4])})
                    Q_evaluations = sess.run(critic.Q_out,feed_dict={critic.state_in: np.vstack(train_batch[:,4])})

                    
                    doubleQ = Q_evaluations[range(batch_size),chosen_actions]
                    targetQ = train_batch[:,1] + gamma * doubleQ
                    
                    _ = sess.run(actor.update_batch,feed_dict={actor.targetQ:targetQ,
                        actor.state_in: np.vstack(train_batch[:,0]), actor.actions: train_batch[:,3]})

                    updateTarget(targetOps,sess)

            if d == True:
                total_reward.append(running_reward)
                total_length.append(j)
                break


        
        #Update our running tally of scores.
        if i % print_freq == 0:
            now = time.time()
            print(i,total_steps,np.mean(total_reward[-100:]),now)
            

        if i % save_freq == 0:
            save_path = saver.save(sess,'saved_models/model'+str(i)+'ckpt')


        i += 1





