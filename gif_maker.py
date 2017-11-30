import imageio
import os
images = []
for p in range(0,2001,100):

	path = 'frames/ep_'+str(p)+'/'
	filenames = os.listdir(path)
	for filename in filenames:
		x = path+filename
		print(x)
		if x != path+'Thumbs.db':
			images.append(imageio.imread(x))

imageio.mimsave('movie.gif', images)