
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import math

### load in image

# image = img.imread('run_track.bmp')[:,:,0]
# image = image == 0
# track_px = image.astype(int)

## define car


class Car():
	def __init__(self,track):
		
		self.track = track
		self.dt = 0.1							# time unit
		self.pos = [448,261]					# start position
		self.theta = 270						# direction pointing (compass)
		self.s = 0								# speed
		self.d = 0								# cuml distance
		self.v = [self.s * math.sin(self.theta*180/math.pi),self.s * math.cos(self.theta*180/math.pi)]

		self.max_turn = 1
		self.max_accel = 1

		self.ray_d = (-90,-60,-30,0,30,60,90)	# direction rays point rel to car dir
		self.ray_l = 50							# length of rays in px
		self.ray_traces = [self.ray_l]*len(self.ray_d)
		self.ray_tracer()

	def reset(self):
		self.pos = [448,261]					# start position
		self.theta = 270						# direction pointing (compass)
		self.s = 0								# speed
		self.d = 0								# cuml distance
		self.v = [self.s * math.sin(self.theta*180/math.pi),self.s * math.cos(self.theta*180/math.pi)]
		self.ray_tracer()
		pass
		
	def ray_tracer(self):

		for i,r in enumerate(self.ray_d):
			on_track = True
			p = 0

			while p < self.ray_l and on_track:
				p += 1
				px = int(np.round(self.pos[0] + p*math.sin((self.theta + r)*math.pi/180)))
				py = int(np.round(self.pos[1] - p*math.cos((self.theta + r)*math.pi/180)))
				on_track = self.track[py,px] == 0
				
			self.ray_traces[i] = p
			pass

	def step(self,action):
		## action comes as a 9-hot encoded vector, (d,0,a) * (l,c,r) = [dl,dc,dr,0l,0c,0r,al,ac,ar]
		
		accel = action // 3 - 1 						# accel and steer now (-1,0,1)
		steer = action % 3 - 1

		self.s += max(accel*self.max_accel*self.dt,0)
		self.theta += steer*self.max_turn
		
		self.v = [self.s * math.sin(self.theta*math.pi/180),self.s * math.cos(self.theta*math.pi/180)]
		self.pos[0] += self.v[0]*self.dt
		self.pos[1] -= self.v[1]*self.dt
		self.d += np.abs(self.s)*self.dt

		self.ray_tracer()

		# check if done
		done = np.amin(self.ray_traces) <= 1 or self.d > 10000 or (self.ray_traces[3])/self.s <= self.dt

		if not done:
			reward = self.s
		elif self.d > 10000:
			reward = 0
		else:
			reward = -20

		return self.pos,self.s,self.v,self.d,self.theta,self.ray_traces,reward,done


###

if __name__ == "__main__":
	
	image = img.imread('run_track.bmp')[:,:,0]
	image = image == 0
	track_px = image.astype(int)

	car = Car(track_px)
	print(car.ray_traces)
	i = 0
	done = False
	while not done and i < 10000000:
		p,s,v,dist,theta,rt,r,done = car.step(7)
		if i % 20 == 0:
			print(p,s,v,dist,theta,rt,car.ray_traces[3]/car.s)
			# plt.matshow(track_px)
			# plt.plot(p[0],p[1],marker='o')
			# plt.show()
		i += 1






