import struct
import numpy as np
import cv2


filename = '101.sift'

list = []

# read the sift file
with open(filename, 'rb') as f:
    aa = 1;
    while(True):
        aa = aa + 1;
        t = f.readline().strip()
        # np.fromstring(t)
        
        if not t or aa > 3 :
            break
        
        # aa = np.frombuffer(a,dtype=a.dtype);
        # a.view(dtype=np.uint8)
        row = np.frombuffer(t, dtype=np.int8)
        print(t)




# header = header.view(dtype=np.uint8)
# print(header)

# location_data = content[20:]

# print(location_data)