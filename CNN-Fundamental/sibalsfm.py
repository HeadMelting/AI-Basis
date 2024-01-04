#SIFT 파일 변환

import cv2
import numpy as np

def Create_SIFT_Match(pathlist):
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptor_list = []
    for index,path in enumerate(pathlist):
        img = cv2.imread()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descriptors = orb.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptor_list.append(descriptors)

        # Pad zeros to make the descriptors 128-dimensional
        des = np.pad(des, ((0, 0), (0, 128 - des.shape[1])), mode='constant')

        # Normalize the descriptors to have a 512 norm
        des = np.array([d / np.linalg.norm(d) * 512 for d in des])

        # Write the keypoints and descriptors to a file in Lowe's ASCII format
        with open(f'{path}.sift', 'w') as f:
            # Write the header
            f.write('{}\n'.format(len(kp)))
            f.write('128\n')  # Descriptor dimensionality
            f.write('# Generated using OpenCV and Python\n')
            
            # Write the keypoints and descriptors
            for i in range(len(kp)):
                x, y = kp[i].pt
                s = kp[i].size
                a = kp[i].angle
                # Scale, angle, and octave are not included in the descriptor,
                # but can be added to the keypoint information if desired.
                desc = ' '.join(['{:0.4f}'.format(x) for x in des[i]])
                f.write('{:.4f} {:.4f} {:.4f} {:.4f} {}\n'.format(x, y, s, a, desc))

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
            f.write(f"{pathlist[i]}.jpg {pathlist[i+1]}.jpg {len(matches)}\n")
            for match in matches:
                f.write(f"{match.queryIdx} ")
            f.write("\n")
            for match in matches:
                f.write(f"{match.trainIdx} ")
            f.write("\n")


ImagePaths = [
    'asdf'
]

Create_SIFT_Match(ImagePaths)





