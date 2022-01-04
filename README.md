# CS386-Deepfake-detection
## 视频流真假人脸对应截取

对于相应的视频流中的真假人脸提取，本项目使用`face_recognition`方法来事先真假人脸的对应截取，具体代码位于`video_cutter`文件夹下

我们的程序默认设置为截取CD2数据集视频，若需截取其他特定文件夹下的对应视频帧中的人脸，请详见`video_cutter`文件夹下的`dataset_generator.py`程序

在`video_cutter`文件夹下下载CD2的数据集后，需先在本地新建以下文件结构：

  Your datast name(此处默认为CD2)
  - fake
  - real

接着输入以下指令生成数据集

```
python dataset_generator.py
```

## 基于SSIM的mask图像生成

若需测试基于SSIM的mask图像生成效果，提供了相应的程序于

## I2G mask图像生成

本项目基于Face-X-ray的开源代码，对于I2G mask 图像生成进行了实现，其具体实现在`X-ray Data Generator`下，我们应我们的需要针对不同视频的不同真实人脸做FaceBlending，在生成真假脸图像的同时，得到其相应的mask图像



## 网络的训练与测试 

### 数据集

基于上述方法得到的用于实际训练与测试的数据集，我们已将其放于交大云盘之上，具体连接如下：

对于其中的.pkl文件，可以将其至于`pkl_data`目录下，通过输入以下指令进行读取，读取到的文件会在训练与测试所用目录`ResNet34+PCL`的`data`与`test_data`文件夹下

```
python dataset_pkl_load.py
```

### 训练

在进行网络的训练之前，我们选定当前目录为`ResNet34+PCL`，训练所用的数据集在`data`文件夹内，接着输入以下指令开始训练

```
python train.py
```

其训练的具体输出将会在产生的`log`文件夹的`__main__.info.log`文件中

另外，训练的具体参数详见 `config.py`文件

### 测试

测试所用的指令与训练相同，测试所用的数据集在`test_data`文件夹内，但需事先将 `config.py`文件中的test、validate或是video_test参数改为测试所用模型的地址，具体详见 `config.py`文件

```
python train.py
```

注意：在本项目中，若test与validate的模型文件地址非空，将优先进行validate与test
