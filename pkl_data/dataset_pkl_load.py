import pickle

from numpy.lib.type_check import real
from tqdm import tqdm
import numpy as np
from PIL import Image

train_load=0
val_load=1
test_load=0

train_path = 'train_'
train_type=['deepfakes','face2face','faceswap','neuraltexture','real']
#train_type=['deepfakes','real']
train_res='.pkl'

val_path='val.pkl'
val_dfd_path='val_dfd.pkl'

test_path='test.pkl'

if train_load:
    li_type=len(train_type)
    id=0
    for t in range(0,li_type):
        f=open(train_path+train_type[t]+train_res,"rb")
        data=pickle.load(f)
        img_list=data['data']
        mask_list=data['target']
        label_list=data['label']

        # print(len(label_list))
        # print(len(mask_list))
        # print(len(img_list))
        li=len(img_list)
        for i in tqdm(range(0,li)):
            id=id+1
            img=Image.fromarray(img_list[i])
            #print(img_list[i].shape)
            #print(mask_list[i])
            mask=np.squeeze(mask_list[i],axis=(2,))
            if label_list[i]==0:
                mask=Image.fromarray(mask,'L')
            else:
                mask=Image.fromarray(mask*0,'L')
            img.save("./data/FF++/train/"+("fake/" if label_list[i]==0 else "real/")+str(id)+("_fake" if label_list[i]==0 else "_real")+'_'+train_type[t]+".jpg")
            mask.save(("./data/FF++/train_mask/"+("fake/" if label_list[i]==0 else "real/")+str(id)+("_fake" if label_list[i]==0 else "_real")+'_'+train_type[t]+".jpg"))

if val_load:
    # f=open(val_path,"rb")
    # data=pickle.load(f)
    # img_list=data['data']
    # label_list=data['label']

    # # print(len(label_list))
    # # print(len(mask_list))
    # # print(len(img_list))
    # id=0
    # li=len(img_list)
    # for i in tqdm(range(0,li)):
    #     id=id+1
    #     img=Image.fromarray(img_list[i])
    #     img.save("./data/FF++/eval/"+("fake/" if label_list[i]==0 else "real/")+str(id)+("_fake" if label_list[i]==0 else "_real")+'_val'+".jpg")
    f=open(val_dfd_path,"rb")
    data=pickle.load(f)
    img_list=data['data']
    label_list=data['label']
    print(0)
    # print(len(label_list))
    # print(len(mask_list))
    # print(len(img_list))
    id=0
    li=len(img_list)
    for i in tqdm(range(0,li)):
        #print(label_list[i])
        id=id+1
        img=Image.fromarray(img_list[i])
        #print(img)
        img.save("./test_data/DFD/"+("fake/" if label_list[i]==0 else "real/")+str(id)+("_fake" if label_list[i]==0 else "_real")+'_test'+".jpg")

if test_load:
    f=open(test_path,"rb")
    data=pickle.load(f)
    img_list=data['data']
    label_list=data['label']
    method_list=data['method']

    DeepF=[]
    Face2F=[]
    FaceS=[]
    NeuralT=[]
    FullFF=[]
    #print(method_list)
    id=0
    li=len(img_list)
    for i in tqdm(range(0,li)):
        data_node=(img_list[i],label_list[i])
        if method_list[i]=='real':
            DeepF.append(data_node)
            Face2F.append(data_node)
            FaceS.append(data_node)
            NeuralT.append(data_node)
            FullFF.append(data_node)
        elif method_list[i]=='Deepfakes':
            DeepF.append(data_node)
            FullFF.append(data_node)
        elif method_list[i]=='Face2Face':
            Face2F.append(data_node)
            FullFF.append(data_node)
        elif method_list[i]=='FaceSwap':
            FaceS.append(data_node)
            FullFF.append(data_node)
        elif method_list[i]=='NeuralTextures':
            NeuralT.append(data_node)
            FullFF.append(data_node)
    print(len(DeepF))
    print(len(Face2F))
    print(len(FaceS))
    print(len(NeuralT))
    print(len(FullFF))
    id=0
    li=len(DeepF)
    for i in tqdm(range(0,li)):
        id=id+1
        img=Image.fromarray(DeepF[i][0])
        img.save("./test_data/DF/"+("fake/" if DeepF[i][1]==0 else "real/")+str(id)+("_fake" if DeepF[i][1]==0 else "_real")+'_test'+".jpg")

    id=0
    li=len(Face2F)
    for i in tqdm(range(0,li)):
        id=id+1
        img=Image.fromarray(Face2F[i][0])
        img.save("./test_data/F2F/"+("fake/" if Face2F[i][1]==0 else "real/")+str(id)+("_fake" if Face2F[i][1]==0 else "_real")+'_test'+".jpg")
    
    id=0
    li=len(FaceS)
    for i in tqdm(range(0,li)):
        id=id+1
        img=Image.fromarray(FaceS[i][0])
        img.save("./test_data/FS/"+("fake/" if FaceS[i][1]==0 else "real/")+str(id)+("_fake" if FaceS[i][1]==0 else "_real")+'_test'+".jpg")

    id=0
    li=len(NeuralT)
    for i in tqdm(range(0,li)):
        id=id+1
        img=Image.fromarray(NeuralT[i][0])
        img.save("./test_data/NT/"+("fake/" if NeuralT[i][1]==0 else "real/")+str(id)+("_fake" if NeuralT[i][1]==0 else "_real")+'_test'+".jpg")
    
    id=0
    li=len(FullFF)
    for i in tqdm(range(0,li)):
        id=id+1
        img=Image.fromarray(FullFF[i][0])
        img.save("./test_data/FF++/"+("fake/" if FullFF[i][1]==0 else "real/")+str(id)+("_fake" if FullFF[i][1]==0 else "_real")+'_test'+".jpg")
