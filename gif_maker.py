import imageio
import os
images = []


# for p in range(0,2001,100):

# 	path = 'frames/ep_'+str(p)+'/'
# 	filenames = os.listdir(path)
# 	for filename in filenames:
# 		x = path+filename
# 		print(x)
# 		if x != path+'Thumbs.db':
# 			images.append(imageio.imread(x))

# imageio.mimsave('movie.gif', images)

run_folders = os.listdir('frames')
run_name = run_folders[-1]
print(run_name)
# run_name = '2017-12-14-22-20-14_12_1e-05_0.6_-0.001'
run_path = 'frames/'+run_name


ep_folders = os.listdir(run_path)

for ep in ep_folders:
	if ep != '.DS_Store':

		path = run_path+'/'+ep+'/'
		filenames = os.listdir(path)
		for filename in filenames:
			x = path+filename
			print(x)
			if x != path+'Thumbs.db':
				images.append(imageio.imread(x))

imageio.mimsave('movie.gif', images,fps=20)