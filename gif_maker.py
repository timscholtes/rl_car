import imageio
import os
images = []
filenames = os.listdir('frames/')
for filename in filenames:
	x = 'frames/'+filename
	print(x)
	if x != 'frames/Thumbs.db':
		images.append(imageio.imread(x))

imageio.mimsave('Z:/GROUP/TIM/GIT/rl_car/movie.gif', images)