import cv2
import face_recognition
from PIL import Image
import numpy as np
import string, random
from tqdm import tqdm
from PIL import Image

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def real_frame_generator(Vid,filename):
    cap = cv2.VideoCapture('./'+filename)
    real_frame_list=[]

    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            real_frame_list.append(frame)
            #cv2.imwrite("./image/fake/"+str(id) + '.jpg', frame)
        else:
            break
    cap.release()
    li = len(real_frame_list)
    sample_list=random.sample(range(0,li),32)
    id=0
    for i in sample_list:
        cv2.imwrite("temp.jpg", real_frame_list[i])
        raw_image = face_recognition.load_image_file("temp.jpg")
        face_locations = face_recognition.face_locations(raw_image)
        if len(face_locations)>0:
            top, right, bottom, left = face_locations[0]
            id = id +1
            raw_face_image = raw_image[top:bottom, left:right]
            pil_image = Image.fromarray(raw_face_image)
            pil_image.save(fp=("./CD2_demo/real/"+str(Vid)+"_"+str(id)+"_real"+".jpg"))
    #print(id)

def fake_frame_generator(Vid,filename):
    
    real_li=filename.split('/')
    vid_name=real_li[1].split('.')
    vid_part=vid_name[0].split('_')

    real_name="./Celeb-real/"+vid_part[0]+'_'+vid_part[2]+'.mp4'
    fake_name='./'+filename

    cap = cv2.VideoCapture(fake_name)
    fake_frame_list=[]
    real_frame_list=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            fake_frame_list.append(frame)
            #cv2.imwrite("./image/fake/"+str(id) + '.jpg', frame)
        else:
            break
    cap.release()

    cap = cv2.VideoCapture(real_name)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            real_frame_list.append(frame)
            #cv2.imwrite("./image/fake/"+str(id) + '.jpg', frame)
        else:
            break
    cap.release()

    li = len(real_frame_list)
    li2 = len(fake_frame_list)
    print(li,li2)
    list_num=list(range(0,li2))
    sample_list=random.sample(list_num,32)
    id=0
    for i in sample_list:
        cv2.imwrite("temp.jpg", real_frame_list[i])
        raw_image = face_recognition.load_image_file("temp.jpg")
        cv2.imwrite("temp.jpg", fake_frame_list[i])
        fake_image = face_recognition.load_image_file("temp.jpg")
        face_locations = face_recognition.face_locations(raw_image)
        if len(face_locations)>0:
            top, right, bottom, left = face_locations[0]
            id = id +1
            face_image = fake_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(fp=("./CD2_demo/fake/"+str(Vid)+"_"+str(id)+"_fake"+".jpg"))
    #print(li,li2)


test_list=[]
with open("List_of_testing_videos.txt", "r") as f:
    test_list= f.readlines()
id=0
li=len(test_list)
for i in tqdm(range(0,li)):
    id=id+1
    line=test_list[i].strip()
    img_pro=line.split(' ')
    if img_pro[0]=='0':
        fake_frame_generator(id,img_pro[1])
    else:
        real_frame_generator(id,img_pro[1])
#print(test_list)