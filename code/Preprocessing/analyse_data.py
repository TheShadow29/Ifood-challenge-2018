import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pdb

no_img = []
max_height = 0
max_width = 0
max_height_img = 'a'
max_width_img = 'a'

img_height = []
img_width = []

for direc in sorted(os.listdir('../../Training_Data/')):
    
    inner_direc = os.path.join('../../Training_Data/',direc)
    no_img.append(len(os.listdir(inner_direc)))
    #pdb.set_trace()
    for images in sorted(os.listdir(inner_direc)):
        print(direc,images)
        img = cv2.imread(os.path.join(inner_direc,images))
        img_height.append(img.shape[0])
        img_width.append(img.shape[1])

        if img.shape[0] > max_height:
            max_height = img.shape[0]
            max_height_img = images
        if img.shape[1] > max_width:
            max_width = img.shape[1]
            max_width_img = images

print("The maximum height is :", max_height)
print("Maximum height image is :",max_height_img)
print("The maximum width is :",max_width)
print("Maximum width image is :",max_width_img)

img_height = np.array(img_height)
mean_img_height = np.mean(img_height)
std_img_height = np.std(img_height)
print("Mean image height :",mean_img_height)
print("Std image height :",std_img_height)

img_width = np.array(img_width)
mean_img_width = np.mean(img_width)
std_img_width = np.std(img_width)
print("Mean image width :",mean_img_width)
print("Std image width :",std_img_width)

no_img = np.array(no_img)
plt.plot(no_img)
plt.title("Image distribution")
plt.xlabel("Classes")
plt.ylabel("No of images")

plt.savefig('histogram.png',dpi=400)

