# CS386-Deepfake-detection
## 视频流真假人脸对应截取
## 基于SSIM的mask图像生成
## I2G mask图像生成
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
