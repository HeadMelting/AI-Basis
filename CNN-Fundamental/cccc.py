import struct
import numpy as np
import cv2

def write_sift_file(filename, location_data, descriptor_data, has_color=False):
    npoint = location_data.shape[0]
    # print(location_data.shape, descriptor_data)
    if has_color:
        name = 0x54494653  # 'SIFT' in little-endian format
        version = 0x305F352E  # '5.0' in little-endian format
    else:
        name = 0x54464953  # 'SIFT' in little-endian format
        version = 0x56352e30  # ('V'+('5'<<8)+('.'<<16)+('0'<<24)) # ColorInfo

    # Create header data
    header = [name, version, npoint, 5, 128]
    header_data = struct.pack('5i', *header)

    # Sort features in the order of decreasing importance
    sorted_indices = np.argsort(-location_data[:, 2])

    # Create location data
    location_data = location_data[sorted_indices, :]
    location_data[:, 2] = location_data[:, 2].clip(0, 255)  # clip color values to [0, 255]
    location_data[:, 3:5] = 0  # set scale and orientation to 0
    location_data = location_data.astype(np.float32)
    location_data = location_data.view(np.uint8).reshape(-1, 20)
    print(location_data.shape)

    ### Create descriptor data
    # Normalize the descriptors to have a 512 norm
    descriptor_data = np.array([d / np.linalg.norm(d) * 512 for d in descriptor_data])
    
    # Pad zeros to make the descriptors 128-dimensional
    padded_descriptor_data = np.pad(descriptor_data, ((0, 0), (0, 128 - descriptor_data.shape[1])), mode='constant')

    padded_descriptor_data = padded_descriptor_data.clip(0, 255)  # clip values to [0, 255]
    padded_descriptor_data = padded_descriptor_data.astype(np.uint8)

    # Write data to file
    with open(filename, 'wb') as f:
        f.write(header_data)
        # print(header.shape)
        f.write(location_data.tobytes())
        print(location_data.shape)
        f.write(padded_descriptor_data.tobytes())
        print(padded_descriptor_data.shape)
        f.write(struct.pack('I', 0xff454f46))   # write EOF marker

def write_matches_txt(descriptor_list,image_names):
     # 특징점 매칭
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_list = []
    for i in range(len(descriptor_list) - 1):
        matches = bf.match(descriptor_list[i], descriptor_list[i+1])
        matches = sorted(matches, key=lambda x: x.distance)
        matches_list.append(matches)

    # 대응 정보를 담은 txt 파일 생성
    with open('matches.txt', 'w') as f:
        for i, matches in enumerate(matches_list):
            f.write(f"{image_names[i]}.jpeg {image_names[i+1]}.jpeg {len(matches)}\n")
            for match in matches:
                f.write(f"{match.queryIdx} ")
            f.write("\n")
            for match in matches:
                f.write(f"{match.trainIdx} ")
            f.write("\n")

def feature_extration_matching(image_paths):
    descriptors_list = []
    image_Names = []

    for path in image_paths:
        imageName = path.split('.')[0]
        image_Names.append(imageName)
        # Load an image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect and compute keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        # Descriptors_list for Matching
        descriptors_list.append(descriptors)

        # Convert keypoints to location_data
        location_data = np.zeros((len(keypoints), 5), dtype=np.float32)
        for i, kp in enumerate(keypoints):
            location_data[i, :] = [kp.pt[0], kp.pt[1], 0, kp.size, kp.angle]

        # Convert descriptors to descriptor_data
        descriptor_data = descriptors.astype(np.float32)

        # Write data to .sift file
        write_sift_file(f'{imageName}.sift', location_data, descriptor_data)

        
    # Write Matches 
    write_matches_txt(descriptors_list,image_Names)
    
image_paths = [
    "K_1.jpeg",
    'K_2.jpeg'
]

feature_extration_matching(image_paths)