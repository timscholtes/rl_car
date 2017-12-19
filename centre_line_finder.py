
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

image = img.imread('runtrack5.bmp')[:,:,0]

print(np.shape(image))
print(type(image))

image = image == 0
track_px = image.astype(int)

# print(track_px[242:280,400])

# y = plt.matshow(track_px)
# plt.show(y)



#####
start_pos = [503,700]


# note that this is y at 0th direction
# note that this is x at 1st direction
def stepper(start_pos, direction = (1,0)):
	start_val = track_px[start_pos[0],start_pos[1]]
	
	next_val = start_val
	# Find edge
	pos = start_pos
	while next_val==start_val:
		pos[0] += direction[0]
		pos[1] += direction[1]

		if pos[0] > np.shape(track_px)[0] or pos[0] < 0:
			return 0
		if pos[1] > np.shape(track_px)[1] or pos[1] < 0:
			return 0

		next_val = track_px[pos[0],pos[1]]

	return pos

pos = stepper(start_pos,direction=(1,0))

if pos == 0:
	print('reversing')
	pos = stepper(start_pos,direction=(-1,0))

print(pos)

##### now for the crawler



def next_edge_finder(start_pos,start_val):
	rotate_displacements = ((1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1))
	# find an adjacent px with the same value as the start pos
	i = -1
	val = 1-start_val
	while val != start_val:
		i += 1
		rot = rotate_displacements[i]
		val = track_px[start_pos[0]+rot[0],start_pos[1]+rot[1]]

	# starting from there, rotate until the value switches. This is then the direction of the edge
	next_val = val
	next_pos = start_pos
	while next_val == val:
		i += 1
		val = next_val
		pos = next_pos
		rot = rotate_displacements[i % 8]
		next_pos = start_pos[0]+rot[0],start_pos[1]+rot[1]
		next_val = track_px[next_pos]

	next_edge_px = pos
	return next_edge_px


def crawler(start_pos):
	start_val = track_px[start_pos[0],start_pos[1]]

	
	pos = next_edge_finder(start_pos,start_val)
	print(pos)
	print(start_pos)
	while pos[0] != start_pos[0] or pos[1] != start_pos[1]:
		pos = next_edge_finder(pos,start_val)
		print(pos)

crawler(start_pos)


	# rotate around the start pos until you find a change of val from 1 to 0

	





		






