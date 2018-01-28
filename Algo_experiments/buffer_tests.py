from collections import deque
import random
import numpy as np
import time
class Experience_buffer_q(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, xp):
        if self.count < self.buffer_size: 
            self.buffer.append(xp)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(xp)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        return np.reshape(np.array(batch),[batch_size,5])
        # s_batch = np.array([_[0] for _ in batch])
        # a_batch = np.array([_[1] for _ in batch])
        # r_batch = np.array([_[2] for _ in batch])
        # t_batch = np.array([_[3] for _ in batch])
        # s2_batch = np.array([_[4] for _ in batch])

        # return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class Experience_buffer_list():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample_batch(self,size):
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
            
    def sample_batch(self,size):
        s = random.sample(self.buffer,size)
        return np.reshape(np.array(s),[size,5])

####

if __name__ == '__main__':

    buff = Experience_buffer_set_list(100000)

    for i in range(100000):
        if i % 100 == 0:
            print(i)
        buff.add([0,0,0,0,0])

    start = time.time()
    for i in range(100000):
        buff.add([0,0,0,0,0])
        if i % 5 == 0:
            buff.sample_batch(100)

        #if i > 50000:
        
    end = time.time()
    print(end-start)

    # adding:

    # sampling

