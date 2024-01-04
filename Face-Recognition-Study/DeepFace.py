## deepface>deepface>DeepFace.py 설명

def build_model():
    """
    Face Identification, Verification Model 선택 및 생성
    """
    return

def verify():
    """
    1. 두 이미지에 대해서 Face Detection을 진행
        - img1_objs, img2_objs 
        - img1_objs = [img1_content, img1_region, ?]
    2. 얼굴만 crop된 영상 (224, 224, 3; pad됨)을 BaseModel(deafult : VGG-Face)에 넣어서 Face representation 진행
        - img1_representation, img2_representation
    3. 구해진 representation간의 거리를 구함
        - cosine, euclidean, euclidean_L2 distance
    """
    return