
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

image = img.imread('runtrack5.bmp')[:,:,0]

print(np.shape(image))
print(type(image))

image = image == 0
track_px = image.astype(int)
print(np.shape(track_px))

# y = plt.matshow(track_px)
# plt.show(y)
