## Blaze Face
## https://developers.google.com/mediapipe/solutions/vision/face_detector/

def build_model():
    import mediapipe as mp

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence = 0.7)
    return face_detection

def detect_face(face_detector, img, align=True):
    resp = []
