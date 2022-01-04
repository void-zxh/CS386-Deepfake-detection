import dlib
import cv2
import os
import json
import numpy as np
import utils as ut
from tqdm import tqdm

"""
Given a set of images, this module allows to compute facial landmarks to
all faces in these images and store them in a json file as a dict.
Given image with unique "ID" (file name), the landmarks will be stored as:
{id1: [[x1,y1], [x2,y2] , ...]}, where the coordinates are points for facial landmarks
in json files
"""

# path for shape_predictor weights for dlib shape_predictor.
# see: https://github.com/davisking/dlib-models
trained_pred_path = ("../shape_predictor_68_face_landmarks.dat")

# path of directory where facial landmarks will be stored
landmark_root_path = "./dataset/landmarks/"
landmark_path="./dataset/landmarks/000/"

# path to image directory
img_dir_root_path="./dataset/images/original/"
img_dir_path = "./dataset/images/original/000/"


class LandmarkDatabaseGenerator():
    def __init__(self, pretrained_pred_p=trained_pred_path,
                 landmark_path=landmark_path,
                 img_dir_path=img_dir_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(trained_pred_path)
        self.imgs = self._get_images()

    def _get_images(self):
        # return list of file names in an image folder
        return [i for i in os.listdir(img_dir_path) if i.lower().endswith((".png", ".jpg", ".jpeg"))]

    def get_landmarks_and_store(self):
        landmarks_dict = {}
        # for each image, get landmarks and store
        for im_p in tqdm(self.imgs):
            # src_im = dlib.load_rgb_image(img_dir_path+im_p)
            src_im = cv2.imread(img_dir_path+im_p)
            b,g,r = cv2.split(src_im)
            src_im = cv2.merge([r,g,b])
            boxes = self.detector(src_im, 1)
            #box = boxes[0]
            box = None
            for box in boxes:
                # take only first detected face in an image
                # since we assume only images of single persons
                box = boxes[0]
                break
            else:
                # skip when no face is found
                continue
            landmarks = self.predictor(src_im, box)
            landmarks_np = ut.shape_to_np(landmarks)
            landmarks_dict[im_p] = landmarks_np.tolist()
        if os.path.exists(landmark_path) == False: #
            os.makedirs(landmark_path)
        with open(landmark_path+'landmark_db.txt', 'w') as outfile:
            json.dump(landmarks_dict, outfile)


def main():
    global landmark_path
    global img_dir_path
    for i in tqdm(range(0,1000)):
        dir=str(i)
        landmark_path=landmark_root_path+dir.zfill(3)+"/"
        img_dir_path=img_dir_root_path+dir.zfill(3)+"/"
        lmGenerator = LandmarkDatabaseGenerator()
        lmGenerator.get_landmarks_and_store()


if __name__ == "__main__":
    main()
