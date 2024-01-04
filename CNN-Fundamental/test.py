import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# set the dimensions of the image
height = 1200
width = 1200
num_bands = 7
pixel_depth = 2

# calculate the total number of pixels in the image
num_pixels = height * width

# open the .bip file
with open('ddd.bip', 'rb') as f:
    # read the file as a byte array
    data = np.fromfile(f, dtype=np.uint16)


# reshape the byte array to a 3D numpy array
image_data = data.reshape(num_pixels, num_bands)
image_data = np.swapaxes(image_data, 0, 1)  # swap the axes to get (bands, pixels, depth)
image_data = image_data.reshape(num_bands, height, width)  # reshape to (bands, height, width, depth)
print(image_data.shape)
image_crop = image_data[:,200:900,200:1000]
print(image_crop[0].shape)
print('최소값: ',np.min(image_crop).shape)

print(np.max(image_data),np.min(image_data))
print(image_data[0,:,:].reshape(-1))

means = []
mins = []
maxs = []
stds = []

for i in range(0,7):
    img = image_crop[i]
    means.append(np.mean(img).round(4))
    mins.append(np.min(img).round(4))
    maxs.append(np.max(img).round(4))
    stds.append(np.std(img).round(4))



#====================================================================================================
#                                               MEAN,STD
#====================================================================================================

# stats = pd.DataFrame({"min":mins,"max":maxs,"mean":means,"std":stds},index=['band1','band2','band3','band4','band5','band6','band7'])
# fig, ax = plt.subplots()
# table = ax.table(cellText=stats.values, colLabels=stats.columns, loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# plt.show()
#=====================================================================================================

# ====================================================================================================
#                                           convert to bsq, bip
# ====================================================================================================
bsq = []
for band in range(0,7):
    for row in range(0,700):
            bsq.extend(image_crop[band,row,:])

bsq = np.array(bsq,dtype=np.uint16)

bip = []
for row in range(0,700):
     for column in range (0,800):
          for band in range(0,7):
               bip.append(image_crop[band,row,column])

bil = []
for row in range(0,700):
    for band in range(0,7):
            bil.extend(image_crop[band,row,:])

bip = np.array(bip,dtype=np.uint16)
bil = np.array(bil,dtype=np.uint16)
bil.tofile('bil2.bil')

bsq.tofile('bsq2.bsq')
bip.tofile('bip2.bip')

# with open('bsq.bsq','w') as f:
#      f.write(bsq)
# f.close()

# with open('bip.bip','w') as f:
#      f.write(bip)
# f.close()
#====================================================================================================

#====================================================================================================
#                                           True Color
#====================================================================================================

# true_color = image_data.transpose(1,2,0)[:,:,0:3]
# true_color = true_color / 65535
# fig = plt.figure(figsize=(2,2))
# fig.add_subplot(1,2,1)
# plt.imshow(true_color)
# plt.title('True Color Composite')
# outlier = true_color >= 0.9
# true_color[outlier] = 0.5
# t_max = np.max(true_color)
# t_min = np.min(true_color)
# brighter_tc = (true_color - t_min)/(t_max - t_min) 
# # brighte_tc = (np.power(((true_color+0.4) * 2),2)/4).clip(0,1)

# fig.add_subplot(1,2,2)
# plt.imshow(brighter_tc)
# plt.title('Brighter Image')
# plt.show()

#====================================================================================================

#====================================================================================================
#                                           image fig
#====================================================================================================
# fig = plt.figure(figsize=(3,3))
# fig.add_subplot(3,3,1)
# plt.title('Band1')
# plt.imshow(image_data[0,:,:])

# fig.add_subplot(3,3,2)
# plt.title('Band2')
# plt.imshow(image_data[1,:,:])

# fig.add_subplot(3,3,3)
# plt.title('Band3')
# plt.imshow(image_data[2,:,:])

# fig.add_subplot(3,3,4)
# plt.title('Band4')
# plt.imshow(image_data[3,:,:])

# fig.add_subplot(3,3,5)
# plt.title('Band5')
# plt.imshow(image_data[4,:,:])

# fig.add_subplot(3,3,6)
# plt.title('Band6')
# plt.imshow(image_data[5,:,:])

# fig.add_subplot(3,3,7)
# plt.title('Band7')
# plt.imshow(image_data[6,:,:])

# plt.show()
#====================================================================================================


#====================================================================================================
#                                                 HISTOGRAM
#====================================================================================================

# fig = plt.figure(figsize=(4,4))
# rows = 2
# columns = 4
# for band in range(0,7):
#     fig.add_subplot(rows,columns,band+1)
#     plt.hist(image_data[band,:,:].reshape(-1),bins=1000,color='g')
#     plt.title(f'Band {band+1}')

# plt.show()
#====================================================================================================