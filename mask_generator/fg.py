import cv2
import face_recognition
from PIL import Image
import numpy as np
import string, random
from tqdm import tqdm

raw_list=[]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def raw_frame_generator(id):
    cap = cv2.VideoCapture("180.mp4")
    global raw_list
    id=0
    while cap.isOpened():
        ret, frame = cap.read()
        id=id+1
        if ret==True:
            #raw_list.append(frame[0:200,0:200])
            cv2.imwrite("./image/raw/"+str(id) + '.jpg', frame)
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def tgt_frame_generator(id):
    cap = cv2.VideoCapture("105.mp4")
    
    id=0
    while cap.isOpened():
        ret, frame = cap.read()
        id = id + 1 
        if ret==True:
            # id=-1
            # for j in range(0,len(raw_list)):
            #     difference = cv2.subtract(frame[0:200,0:200],raw_list[j])
            #     result = not np.any(difference)
            #     if result is True:
            #         id = j+1
            #         break
            # if id !=-1:
            cv2.imwrite("./image/fake/"+str(id) + '.jpg', frame)
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return id

raw_frame_generator("000")
num=tgt_frame_generator("000")
cou=0
for i in tqdm(range(1,num)):
    raw_image = face_recognition.load_image_file("./image/raw/{}.jpg".format(i))
    fake_image = face_recognition.load_image_file("./image/fake/{}.jpg".format(i))
    face_locations = face_recognition.face_locations(raw_image)
    #face_locations 以列表形式返回图片中的所有人脸

    for fi in face_locations:
        top, right, bottom, left = fi
        cou = cou +1
        raw_face_image = raw_image[top:bottom, left:right]
        pil_image = Image.fromarray(raw_face_image)
        pil_image.save(fp="./data/raw/{}.jpg".format(cou))
    
    face_locations = face_recognition.face_locations(fake_image)
    
    for fi in face_locations:
        top, right, bottom, left = fi
        fake_face_image = fake_image[top:bottom, left:right]
        pil_image = Image.fromarray(fake_face_image)
        pil_image.save(fp="./data/fake/{}.jpg".format(cou))
