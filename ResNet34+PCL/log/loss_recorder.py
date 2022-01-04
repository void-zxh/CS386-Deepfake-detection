
import matplotlib.pyplot as plt
import numpy as np

def record_losses(input):
    f=open(input,encoding='utf-8')
    re_list=[]
    for line in f:
        li_str=str(line)
        if li_str.find("losses")!=-1:
            id=li_str.find("losses")
            re_list.append(float(li_str[id+8:-1]))
    print(re_list)
    return re_list


if __name__=="__main__":
    losses_list1=record_losses('__main__.info_pcl+I2G.log')
    print(len(losses_list1))
    losses_list2=record_losses('__main__.info_pcl10.log')
    print(len(losses_list2))
    x=np.arange(1,101,1)
    plt.plot(x,losses_list1,label='pcl+I2G')
    plt.plot(x,losses_list2,label='pcl')
    plt.legend(['pcl+I2G','pcl'])
    plt.xlabel("Epoch") 
    plt.ylabel("Losses")
    plt.show()