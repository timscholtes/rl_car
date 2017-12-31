
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec 
import numpy as np
import math
import os
### load in image

# image = img.imread('run_track.bmp')[:,:,0]
# image = image == 0
# track_px = image.astype(int)

## define car


class Car():
	def __init__(self,track,r_scale):
		
		self.track = track
		#self.track = np.pad(self.track,(200,200),'constant',constant_values=[1,1])
		self.r_scale = r_scale
		self.dt = 0.15							# time unit
		self.pos = [700,503]#[448,261]					# start position
		self.theta = 270						# direction pointing (compass)
		self.s = 10								# speed
		self.d = 0								# cuml distance
		self.v = [self.s * math.sin(self.theta*180/math.pi),self.s * math.cos(self.theta*180/math.pi)]

		self.max_turn = 2
		self.max_accel = 1
		self.max_speed = 40.0
		self.min_speed = 0.0

		self.ray_d = (-60,-40,-20,0,20,40,60)	# direction rays point rel to car dir
		#self.ray_d = (-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60)	# direction rays point rel to car dir
		self.ray_l = 200.0							# length of rays in px
		self.ray_traces = [self.ray_l]*len(self.ray_d)
		self.ray_tracer()
		self.speed_history = [self.s]
		self.dist_history=[0]
		self.prev_speed_dist_histories = []

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
		self.speed_history = [self.s]
		self.dist_history=[0]
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

	def step(self,action,recording=False):
		## action comes as a length 2 tuple, of numbers between -1,1 for how far to accel and steer
		
		accel = action[0]
		steer = action[1]

		self.theta += steer*self.max_turn
		self.s += accel*self.max_accel
		# self.s = min(max(self.min_speed,self.s),self.max_speed)
		self.s = min(self.s,self.max_speed)
		
		self.v = [self.s * math.sin(self.theta*math.pi/180),self.s * math.cos(self.theta*math.pi/180)]
		self.pos[0] += self.v[0]*self.dt
		self.pos[1] -= self.v[1]*self.dt
		self.d += np.abs(self.s)*self.dt

		self.ray_tracer()

		if recording:
			self.speed_history.append(self.s)
			self.dist_history.append(self.d)

		# check if done
		done = np.amin(self.ray_traces) <= 1 or self.d > 10000 or (self.ray_traces[3])/(self.s+0.01) <= self.dt

		if not done:
			reward = self.r_scale*self.s/self.max_speed
		elif self.d > 10000:
			reward = 0
		else:
			reward = -1

		if done and recording:
			self.prev_speed_dist_histories.append([self.speed_history,self.dist_history])

		state = self.ray_traces
		state = [i / self.ray_l for i in state]
		state.append(self.s/self.max_speed)
		return state,reward,done

	def plotter(self,i,title,save_dest,extra=None):
		fig = plt.figure(figsize=(15,4))
		fig.suptitle(title, fontsize=16)
		ax1 = plt.subplot2grid((1,7),(0,0),colspan=2,rowspan=2)
		ax2 = plt.subplot2grid((1,7),(0,2))
		ax3 = plt.subplot2grid((1,7),(0,3))
		ax4 = plt.subplot2grid((1,7),(0,4))
		ax5 = plt.subplot2grid((1,7),(0,5))
		ax6 = plt.subplot2grid((1,7),(0,6))

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

		ax5.set_ylim([-1,1])
		ax5.bar([0],extra[0])
		ax5.axhline(y=0)
		ax5.xaxis.set_ticklabels([])

		ax6.set_xlim([-1,1])
		ax6.barh([-1,0,1],[0,extra[1],0])
		ax6.axvline(x=0)
		ax6.yaxis.set_ticklabels([])

		plt.savefig(save_dest)
		plt.close('all')

	def plotter2(self,i,title,save_dest,extra=None):
		
		delta = 0.01
		x = np.arange(-1,1,delta)
		y = np.arange(-1,1,delta)
		X,Y = np.meshgrid(x,y)
		Z = mlab.bivariate_normal(X,Y,sigmay=extra[2],sigmax=extra[3],muy=extra[0],mux=extra[1])

		fig = plt.figure(figsize=(16,8))
		fig.suptitle(title, fontsize=12)

		ax1 = plt.subplot2grid((2,6),(0,0),colspan=3)
		ax2 = plt.subplot2grid((2,6),(0,3),colspan=3)
		ax3 = plt.subplot2grid((2,6),(1,0),colspan=3)
		ax4 = plt.subplot2grid((2,6),(1,3))
		ax5 = plt.subplot2grid((2,6),(1,4),projection='polar',colspan=2)

		all_track = 1-self.track
		zoom_wd = 250

		miny = max(self.pos[1]-zoom_wd,0)
		maxy = min((self.pos[1]+zoom_wd),np.shape(all_track)[0])
		minx = max((self.pos[0]-zoom_wd),0)
		maxx = min((self.pos[0]+zoom_wd),np.shape(all_track)[1])

		zoom_track = all_track[miny:maxy,minx:maxx]

		repos = [self.pos[0]-minx,self.pos[1]-miny]
		ax1.imshow(zoom_track,cmap='gray')
		ax1.plot(repos[0],repos[1],marker='.')
		dx = max(self.s,1)*10*self.dt*math.sin(self.theta*math.pi/180)
		dy = -max(self.s,1)*10*self.dt*math.cos(self.theta*math.pi/180)
		
		ax1.arrow(repos[0],repos[1],dx = dx,dy=dy,shape='full',head_width=5)
		ax1.autoscale()
		# ax1.yaxis.set_ticklabels([])
		# ax1.xaxis.set_ticklabels([])
		
		ax2.set_ylim(-1,1)
		ax2.set_xlim(-1,1)
		ax2.contour(X,Y,Z,1,colors='k')
		ax2.plot(extra[1],extra[0],'ro')
		ax2.axvline(x=0)
		ax2.axhline(y=0)
		ax2.yaxis.set_ticklabels([])
		ax2.xaxis.set_ticklabels([])
		ax2.set_title('Accel/Steer Distribution')
		ax2.set_xlabel('Steer: Left/Right')
		ax2.set_ylabel('Acceleration')

		# ax3.set_ylim([0,self.max_speed+10])
		if len(self.prev_speed_dist_histories) != 0:
			for p in self.prev_speed_dist_histories:
				ax3.plot(p[1],p[0],color='lightgrey')
		ax3.plot(self.dist_history,self.speed_history)
		ax3.set_xlabel('Distance')
		ax3.set_ylabel('Speed')

		ax4.set_ylim([-10,self.max_speed+10])
		ax4.set_title('Speed')
		ax4.bar(1,self.s)
		ax4.xaxis.set_ticklabels([])
		

		ax5.set_ylim([0,self.ray_l])
		# radii = [-(j+self.theta)*np.pi/180 for j in self.ray_d]
		radii = [j*np.pi/180 for j in self.ray_d]
		ax5.bar(radii,self.ray_traces,width=[20*np.pi/180]*len(self.ray_d))

		ax5.set_title('Ray Distances')
		ax5.set_theta_zero_location('N')
		ax5.set_thetamin(-60)
		ax5.set_thetamax(60)
		ax5.set_theta_direction(-1)

		ax5.set_yticklabels([])
		ax5.yaxis.grid(False)
		ax5.xaxis.grid(False)
		r = [0,self.ray_l]
		theta = [extra[1]*self.max_turn*np.pi*20/180]*len(r)
		ax5.plot(theta,r,lw=1,color='red')
		

		plt.savefig(save_dest)
		plt.close('all')



###

if __name__ == "__main__":

	image = img.imread('runtrack5.bmp')[:,:,0]
	image = image == 0
	track_px = image.astype(int)
	print(np.shape(track_px))

	car = Car(track_px,0.1)
	# car.pos[0] = 80
	# car.pos[1] = 450
	i = 0
	done = False
	if not os.path.exists('./frames/tmp/'):
		os.makedirs('./frames/tmp/')

	running_r = 0
	car.plotter2(i,'abc','frames/tmp/'+str(i).zfill(4)+'.png',[0.3,-1,0.01,0.3])

	# while not done and i < 1000:
	# 	#move = (np.random.random()*2-1,np.random.random()*2-1)
	# 	# state,r,done = car.step(np.random.randint(3))
	# 	state,r,done = car.step(move)
	# 	running_r += r
	# 	if i % 20 == 0:
	# 		print(i)

	# 		car.plotter(i,'abc','frames/tmp/'+str(i).zfill(4)+'.png',move)
	# 	i += 1
	# print(running_r)






