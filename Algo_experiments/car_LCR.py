
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import math
import os
### load in image

# image = img.imread('run_track.bmp')[:,:,0]
# image = image == 0
# track_px = image.astype(int)

## define car


class Car():
	def __init__(self,track):
		
		self.track = track
		self.dt = 0.15							# time unit
		self.pos = [700,503]#[448,261]					# start position
		self.theta = 270						# direction pointing (compass)
		self.s = 10								# speed
		self.d = 0								# cuml distance
		self.v = [self.s * math.sin(self.theta*180/math.pi),self.s * math.cos(self.theta*180/math.pi)]

		self.max_turn = 1
		self.max_accel = 1
		self.max_speed = 50.0

		self.ray_d = (-60,-40,-20,0,20,40,60)	# direction rays point rel to car dir
		self.ray_l = 100.0							# length of rays in px
		self.ray_traces = [self.ray_l]*len(self.ray_d)
		self.ray_tracer()

	def reset(self):
		self.pos = [700,503]#[448,261]					# start position
		self.theta = 270						# direction pointing (compass)
		self.s = 10								# speed
		self.d = 0								# cuml distance
		self.v = [self.s * math.sin(self.theta*180/math.pi),self.s * math.cos(self.theta*180/math.pi)]
		self.ray_tracer()
		state = self.ray_traces
		state = [i / self.ray_l for i in state]
		state.append(self.s/self.max_speed)
		return state,0,False
		
	def ray_tracer(self):

		for i,r in enumerate(self.ray_d):
			on_track = True
			p = 0

			while p <= self.ray_l and on_track:
				if p <= 20:
					p += 1
				elif p < 50:
					p += 5
				else:
					p += 10
				px = int(np.round(self.pos[0] + p*math.sin((self.theta + r)*math.pi/180)))
				py = int(np.round(self.pos[1] - p*math.cos((self.theta + r)*math.pi/180)))
				on_track = self.track[py,px] == 0
				
			self.ray_traces[i] = p
			pass

	def step(self,action):
		## action comes as a 9-hot encoded vector, (d,0,a) * (l,c,r) = [dl,dc,dr,0l,0c,0r,al,ac,ar]
		## NOW action comes as a number 0:2, either L,C,R
		
		steer = action -2

		self.theta += steer*self.max_turn
		
		self.v = [self.s * math.sin(self.theta*math.pi/180),self.s * math.cos(self.theta*math.pi/180)]
		self.pos[0] += self.v[0]*self.dt
		self.pos[1] -= self.v[1]*self.dt
		self.d += np.abs(self.s)*self.dt

		self.ray_tracer()

		# check if done
		done = np.amin(self.ray_traces) <= 1 or self.d > 10000 or (self.ray_traces[3])/(self.s+0.01) <= self.dt

		if not done:
			reward = 0 #self.s/self.max_speed
		elif self.d > 10000:
			reward = 0
		else:
			reward = -1

		state = self.ray_traces
		state = [i / self.ray_l for i in state]
		state.append(self.s/self.max_speed)
		return state,reward,done

	def plotter(self,i,title,save_dest,extra=None):
		fig = plt.figure(figsize=(15,4))
		fig.suptitle(title, fontsize=16)
		ax1 = plt.subplot2grid((1,6),(0,0),colspan=2,rowspan=2)
		ax2 = plt.subplot2grid((1,6),(0,2))
		ax3 = plt.subplot2grid((1,6),(0,3))
		ax4 = plt.subplot2grid((1,6),(0,4))
		ax5 = plt.subplot2grid((1,6),(0,5))

		ax1.imshow(1-self.track,cmap='gray')
		ax1.plot(self.pos[0],self.pos[1],marker='.')
		dx = max(self.s,1)*10*self.dt*math.sin(self.theta*math.pi/180)
		dy = -max(self.s,1)*10*self.dt*math.cos(self.theta*math.pi/180)
		
		ax1.arrow(self.pos[0],self.pos[1],dx = dx,dy=dy,shape='full',head_width=5)
		ax1.autoscale()
		
		ax2.set_ylim([0,50])
		ax2.set_title('Speed')
		ax2.bar(1,self.s)

		ax3.set_ylim([0,self.ray_l])
		ax3.bar([-60,-40,-20,0,20,40,60],self.ray_traces,width=20)

		ax4.text(0.1,0.5,'Distance:'+str(np.round(self.d)))
		ax4.text(0.1,0.8,'Timestep:'+str(i))
		ax4.axis('off')

		ax5.set_ylim([0,1])
		ax5.bar([-2,-1,0,1,2],extra,width=1)

		plt.savefig(save_dest)
		plt.close('all')

###

if __name__ == "__main__":

	image = img.imread('runtrack5.bmp')[:,:,0]
	image = image == 0
	track_px = image.astype(int)
	
	car = Car(track_px)

	
	i = 0
	done = False
	os.makedirs('frames/tmp/')
	running_r = 0
	while not done and i < 10000000:
		if i < 140:
			move = 1
		else:
			move = 2
		# state,r,done = car.step(np.random.randint(3))
		state,r,done = car.step(move)
		running_r += r
		if i % 20 == 0:
			print(i)

			car.plotter(i,'abc','frames/tmp'+str(i).zfill(4)+'.png',[0.1,0.5,0.4])
		i += 1
	print(running_r)






