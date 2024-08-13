import cv2
import numpy as np
import math
# import dlib
# import face_alignment

class Landmark_helper:
    def __init__(self, Method_type='Efficient',num_pts=68):
        self.method_type=Method_type
        self.predictor=None
        self.shapes=None
        self.num_pts=num_pts
        self.bbox=None

    def detect_facelandmark(self,image_color):
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) > 0 :
            if self.method_type=='dlib':
                self.predictor = dlib.shape_predictor(
                    "./model/68_face_landmarks.dat")
                shape = self.predictor(gray, rects[0])
                self.bbox=self.rect_to_bb(rects[0])
                shapes=[]
                for i in range(self.num_pts):
                    shapes.append(shape.parts()[i].x)
                    shapes.append(shape.parts()[i].y)
                self.shapes=np.asarray(shapes).reshape(-1,2)
            elif self.method_type=='hourglass':
                fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cuda:0')
                preds = fa.get_landmarks_from_image(image_color)
                if preds:
                    self.shapes=np.asarray(preds).reshape(-1,2)
            else:
                bbox=self.rect_to_bb(rects[0])
                new_bbox=self.enlarge_bbox(gray,bbox)
            face_exist = True
        else:
            self.shapes = None
            face_exist = False

        return self.shapes, face_exist

    def Draw(self,img):
        if self.method_type=='dlib':
            # cv2.rectangle(img,(self.bbox[0],self.bbox[1]),(self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]),(0,0,255),5)
            for pt in range(self.num_pts):
                pt_pos = (self.shapes[pt][0], self.shapes[pt][1])
                cv2.circle(img, pt_pos, 5, (0, 255,0), -1)
            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("image", img)
            # cv2.imwrite('4.png',img)
            # cv2.waitKey()
        elif self.method_type=='hourglass':
            print(self.shapes.shape)
            for pt in range(self.num_pts):
                pt_pos = (self.shapes[pt][0], self.shapes[pt][1])
                cv2.circle(img, pt_pos, 5, (0, 255,0), -1)
                # cv2.imwrite('res.png', img)

    def rect_to_bb(self,rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    def enlarge_bbox(image,bbox, ratio=0.1):
        assert len(bbox) == 4, 'bbox.shape is {}'.format(bbox.shape)
        if ratio <= 0.0:
            return bbox
        width = bbox[2]
        height = bbox[3]

        newbbox = np.zeros_like(bbox)
        newbbox[0] = np.min(0,bbox[0] - ratio * width)
        newbbox[1] = np.min(0,bbox[1] - ratio * height)
        newbbox[2]=np.max(image.size[0],math.ceil((ratio*2+1.0)*width))
        newbbox[3]=np.max(image.size[1],math.ceil((ratio*2+1.0)*height))
        return newbbox

def getbbox(landmarks):
    landmarks=landmarks.reshape(-1,2)
    xmin=np.min(landmarks[:,0])
    xmax=np.max(landmarks[:,0])
    ymin=np.min(landmarks[:,1])
    ymax=np.max(landmarks[:,1])
    width=math.ceil(xmax-xmin)
    height=math.ceil(ymax-ymin)

    cen_x=xmin+width/2
    cen_y=ymin+height/2
    if width==height:
        return [xmin,ymin,xmin+width,ymin+height]
    else:
        if width>height:
            ymin=cen_y-width/2
            return [xmin,ymin,xmin+width,ymin+width]
        elif height>width:
            xmin=cen_x-height/2
            return [xmin,ymin,xmin+height,ymin+height]

def crop_landmarkbox(landmarks,ratio=1/16.0):
    bbox=getbbox(landmarks)
    # print(bbox)
    assert len(bbox)==4,'bbox.shape is {}'.format(bbox.shape)
    if ratio<0.0:
        return bbox
    width=bbox[2]-bbox[0]
    height=bbox[3]-bbox[1]

    newbbox=np.zeros_like(bbox,dtype=int)
    extension=ratio*(width+height)
    newbbox[0]=int(bbox[0]-extension)
    newbbox[1]=int(bbox[1]-extension)
    newbbox[2] = int(bbox[2]+extension)
    newbbox[3] = int(bbox[3]+extension)
    return newbbox
    
def transform_pts(pt, M, invert=0):
    # Transform pixel location to different reference
    if invert:
        M = np.linalg.inv(M)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(M, new_pt)
    return new_pt[:2].astype(int) + 1

def save_image(image,landmarks,name):
    image_bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for pt in range(landmarks.shape[0]):
        pt_pos = (int(landmarks[pt][0]), int(landmarks[pt][1]))
        cv2.circle(image_bgr, pt_pos, 5, (0, 255,0), -1)
        # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imwrite(name,image_bgr)


REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]
point_index=[36,45,33,48,54]

def align_process(img, landmarks, image_size):
    """
    crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256 (Height,width)
        padding: default 0
    Retures:
    -------
    crop_imgs: list, n
        cropped and aligned faces
    """
    M = None
    if landmarks is not None:
        assert len(image_size) == 2
        # 这个基准是112*96的面部特征点的坐标
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if image_size[0]!=112:
            src[0]*=image_size[0]/112
        if image_size[1]!=96:
            src[1]*=image_size[1]/96

        landmarks = landmarks.astype(np.float32)
        dst=np.zeros((5,2),dtype=float)

        for i in range(5):
            dst[i]=landmarks[point_index[i]]

        M, _ = cv2.estimateAffine2D(dst, src)

        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        landmark_new = np.array (landmarks, dtype=np.float32)
        for i in range (landmarks.shape[0]):
            landmark_new[i]=transform_pts(landmark_new[i],M)

        return warped,landmark_new


def align_face(image_array,landmarks,method_type='similar'):
    # get list landmarks of left and right eye
    left_eye = landmarks[36]
    right_eye =landmarks[45]
    # center=np.zeros_like(left_eye)
    # center[0]=(left_eye[0]+right_eye[0])//2
    # center[1]=(left_eye[1]+right_eye[1])//2
    center = landmarks[30]
    if method_type=='similar':
        # compute the angle between the eye centroids
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        # compute angle between the line of 2 centeroids and the horizontal line
        angle = math.atan2(dy, dx) * 180. / math.pi
        # at the eye_center, rotate the image by the angle
        M = cv2.getRotationMatrix2D((center[0],center[1]), angle, scale=1)
        rotated_img = cv2.warpAffine(image_array, M,
                                        (image_array.shape[1], image_array.shape[0]))
        landmark_new = np.array (landmarks, dtype=np.float32)
        for i in range (landmarks.shape[0]):
            landmark_new[i]=transform_pts(landmark_new[i],M)
        return rotated_img, landmark_new



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)   #设精度
    test=Landmark_helper(Method_type='hourglass')
    Dirs=np.loadtxt('/home/gpc/FaceAU/Dataset/ImageDir_List2.txt',dtype=str)
    data_root='/HDD/gpc/feafa+'
    for dir in Dirs:
        dir=os.path.join(data_root,dir)
        image_paths=[i for i in os.listdir(dir) if i[-3:] in ['png','jpg']]
        image_paths.sort(key=lambda x: str(x.split('_')[-1]))
        test.shapes=None
        for image_path in image_paths:
            image_path=os.path.join(dir,image_path)
            image=cv2.imread(image_path)
            if image is not None:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                test.detect_facelandmark(image)
                save_path=image_path.replace('output','landmark_hourglass')
                save_path=save_path.replace('png', 'txt')
                save_path=save_path.replace('jpg', 'txt')
                print(save_path)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if test.shapes is not None:
                    np.savetxt(save_path,test.shapes,fmt='%.04f')
                else:
                    print('error ......_______>>>>>>>')