import dlib
import cv2
import numpy as np
def align_face(image):
    # 初始化dlib的人脸检测器和关键点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('68_face_landmarks.dat')
    import ipdb;ipdb.set_trace()
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用dlib的人脸检测器检测人脸
    faces = detector(gray)

    # 如果检测到多个人脸，选择最大的人脸
    if len(faces) > 0:
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # 使用关键点检测器获取人脸关键点
        shape = predictor(gray, face)

        # 将关键点坐标转换为numpy数组
        landmarks = np.array([[p.x, p.y] for p in shape.parts()]).astype('float64')

        # # 定义对齐目标点坐标
        # align_landmarks = np.array([(127.5, 160.5), (382.5, 160.5), (255.0, 255.0)])
        # align_landmarks = np.array([(55.5, 71.5), (168.5, 71.5), (112.0, 112.0)])

        # # 使用仿射变换将人脸对齐到目标点
        # transform = cv2.getAffineTransform(landmarks[:3], align_landmarks)
        # aligned_face = cv2.warpAffine(image, transform, (224, 224))

        return landmarks

    else:
        # 如果未检测到人脸，则返回原始图像
        return image
    
