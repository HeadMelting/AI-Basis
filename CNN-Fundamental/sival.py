import cv2
import numpy as np

# 이미지 파일 경로 리스트
image_paths = [
    'C:/YJpython/sensor_paper/K_1.jpg',
    'C:/YJpython/sensor_paper/K_2.jpg',
    'C:/YJpython/sensor_paper/K_3.jpg',
    'C:/YJpython/sensor_paper/K_4.jpg',
    'C:/YJpython/sensor_paper/K_5.jpg',
    'C:/YJpython/sensor_paper/K_6.jpg',
    'C:/YJpython/sensor_paper/K_7.jpg',
    'C:/YJpython/sensor_paper/K_8.jpg',
    'C:/YJpython/sensor_paper/K_9.jpg',
    'C:/YJpython/sensor_paper/K_10.jpg'
]

#특징점 검출 알고리즘을 적용하여 특징점 위치를 추출하고 두 이미지 간에 대응되는 특징점 쌍을 찾아내는 작업

# ORB 객체 생성
orb = cv2.ORB_create()
# 이미지에서 특징점 검출 및 기술자 계산
keypoints_list = []
descriptor_list = []
for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoints_list.append(keypoints)
    descriptor_list.append(descriptors)

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
        f.write(f"{i}.jpg {i+1}.jpg {len(matches)}\n")
        for match in matches:
            f.write(f"{match.queryIdx} {match.trainIdx}\n")