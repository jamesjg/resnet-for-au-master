import torch
import numpy as np
import argparse
import os
import cv2
from mtcnn import MTCNN
import numpy as np


class FaceDetecter():
    def __init__(self, net_type="mtcnn", return_type="v1", scale=1.0):
        self.net_type = net_type
        self.return_type = return_type
        self.scale = scale

        # init network according the net_type
        if self.net_type == "mtcnn":
            self.net = MTCNN()
        else:
            # Initialize RetinaNet or other face detection models
            pass

    def detect_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect face boxes
        boxes = self.net.detect_faces(img)

        # filter useless boxes with rules
        filtered_boxes = self.filter_boxes(boxes,img)
        # adjust box size according max width or height (maintain the wh ratio)
        adjusted_boxes = self.adjust_box_size(filtered_boxes, img.shape[1], img.shape[0])
        box_info = self.convert_to_return_type(adjusted_boxes)
        return box_info

        # enlarge box size when scale > 1.0
        #enlarged_boxes = self.enlarge_box_with_scale(adjusted_boxes)

        # return box information by return_type

    def filter_boxes(self, boxes, img):
        # Filter face boxes based on specific rules
        filtered_boxes = []

        if len(boxes) != 0:

            box_areas = np.array([box['box'][2] * box['box'][3] for box in boxes]) / (img.shape[0] * img.shape[1])
            center_points = np.array(
                [(box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2) for box in boxes])
            center_differences = np.linalg.norm(center_points - np.array(img.shape[:2])[::-1] // 2, axis=1)
            confidences = np.array([box['confidence'] for box in boxes])
            idx_center_small = np.argsort(center_differences)
            idx_area_large = np.argsort(box_areas)[::-1]

            if idx_center_small[0] != idx_area_large[0]:
                if box_areas[idx_center_small[0]] > 0.15 and confidences[idx_center_small[0]] > 0.95:
                    max_largest_idx = idx_center_small[0]
                else:
                    max_largest_idx = idx_area_large[0]
            else:
                max_largest_idx = idx_center_small[0]

            filtered_boxes.append(boxes[max_largest_idx])


        else:

            height, width, _ = img.shape
            filtered_boxes.append({
                'box': [0, 0, width, height],  # 设置为图像边角的四个顶点坐标
                'confidence': 0.0,  # 设置置信度为0
                'keypoints': {}  # 设置空的关键点
            })

        return filtered_boxes

    # def adjust_box_size(self, boxes, max_width, max_height):
    #     adjusted_boxes = []
    #     for box in boxes:
    #         center_point = (box['box'][0] + box['box'][2]//2, box['box'][1] + box['box'][3]//2)
    #         edge_length = max(box['box'][2], box['box'][3])
    #         new_size = (int(edge_length * self.scale), int(edge_length * self.scale))
    #         right_bottom = (center_point[0] + new_size[0]//2, center_point[1] + new_size[1]//2)
    #         left_top = (center_point[0] - new_size[0]//2, center_point[1] - new_size[1]//2)
    #         right_bottom = (min(right_bottom[0], max_width), min(right_bottom[1], max_height))
    #         left_top = (max(left_top[0], 0), max(left_top[1], 0))
    #         adjusted_boxes.append({'box': (left_top[0], left_top[1], right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]),
    #                                'confidence': box['confidence']})
    #     return adjusted_boxes




    def adjust_box_size(self, boxes, max_width, max_height):
        adjusted_boxes = []
        for box in boxes:
            box_info = box['box'] if 'box' in box else None
            if box_info is None or len(box_info) != 4:
                continue

            confidence = box['confidence']

            center_point = (int(box_info[0] + box_info[2] // 2), int(box_info[1] + box_info[3] // 2))

            edge_length = max(box_info[2], box_info[3])
            new_size = (int(edge_length * self.scale), int(edge_length * self.scale))
            right_bottom = (center_point[0] + new_size[0] // 2, center_point[1] + new_size[1] // 2)
            left_top = (center_point[0] - new_size[0] // 2, center_point[1] - new_size[1] // 2)
            right_bottom = (min(right_bottom[0], max_width), min(right_bottom[1], max_height))
            left_top = (max(left_top[0], 0), max(left_top[1], 0))
            adjusted_boxes.append(
                {'box': (left_top[0], left_top[1], right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]),
                 'confidence': confidence})
        return adjusted_boxes

    # def enlarge_box_with_scale(self, boxes):
    #     # Enlarge box size when scale > 1.0
    #     enlarged_boxes = []
    #     # Implement your box enlargement logic here
    #     return enlarged_boxes

    def convert_to_return_type(self, adjusted_boxes):
        box_info = []

        if self.return_type == "v1":
            # Convert to return type v1: left-top point and width, height
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (left_top, width, height)
            box_info = (box['box'][0], box['box'][1], width, height)


        elif self.return_type == "v2":
            # Convert to return type v2: left-top and right-bottom points
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            right_bottom = (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])
            # box_info = (left_top, right_bottom)
            box_info = (box['box'][0], box['box'][1],box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])

        elif self.return_type == "v3":
            # Convert to return type v3: center point and width, height
            box = adjusted_boxes[0]
            center = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2)
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (center, width, height)
            box_info = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2,box['box'][2], box['box'][3])

        print(box_info)

        return box_info

    def vis_result(self, img_path, box_info):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Visualize the result on the image
        vis_img = img.copy()

        # Draw the detected face boxes on the image
        left,top, width, height = box_info  # Assuming box_info is in the format (left_top, width, height)
        # right_bottom = (left_top[0] + width, left_top[1] + height)
        right_bottom = (left + width, top + height)
        left_top=(left,top)
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)

        # Save the visualization image
        output_dir = os.path.join(os.path.dirname(img_path), "vis")
        os.makedirs(output_dir, exist_ok=True)

        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{img_name}_vis")
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)  # Correct color space conversion
        cv2.imwrite(output_path, vis_img)


    # #输入图片路径，保存检测结果（box_info)
    # def detect_folder(self, img_folder):
    #     for filename in os.listdir(img_folder):
    #         if filename.endswith(".jpg") or filename.endswith(".png"):
    #             img_path = os.path.join(img_folder, filename)
    #             box_info = self.detect_img(img_path)
    #             self.vis_result(img_path, box_info)


    def detect_folder(self, img_folder):
        output_filename = f"output_{args.scale}_{args.net_type}_{args.return_type}.txt"  # 构造输出文件名
        output_path = os.path.join(img_folder, output_filename)  # 构造输出文件路径
        with open(output_path, 'w') as f:
            for filename in os.listdir(img_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(img_folder, filename)
                    box_info = self.detect_img(img_path)
                    f.write(f"{img_path} {box_info}\n")
                    self.vis_result(img_path, box_info)
        return box_info


    def process_images(self, img_paths):
        results = []
        for img_path in img_paths:
            box_info = self.detect_img(img_path)
            results.append(box_info)
            print(results)
        return results





def parse_args():
    parser = argparse.ArgumentParser(description='Face Detecting')
    parser.add_argument('--net_type', default="mtcnn", type=str, choices=["mtcnn", "retinaface"], help='choose network')
    parser.add_argument('--return_type', default="v1", type=str, choices=["v1", "v2", "v3", "v4"],
                        help='choose return type')
    parser.add_argument('--scale', default=1.0, type=float, help='the scale of box size')
    parser.add_argument('--detect_folder', default=True, action='store_true', help="detect folder or img file")
    parser.add_argument('--path', required=True, type=str, help="img file path or folder path")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    detector = FaceDetecter(args.net_type, args.return_type, args.scale)
    if args.detect_folder:
        info=detector.detect_folder(args.path)
    else:
        info=detector.detect_img(args.path)
        detector.vis_result(args.path, info)


