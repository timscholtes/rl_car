
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

image = img.imread('run_track.bmp')[:,:,0]

print(np.shape(image))
print(type(image))

image = image == 0
track_px = image.astype(int)

# print(track_px[242:280,400])

y = plt.matshow(track_px)
plt.show(y)
