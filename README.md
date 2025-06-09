# Iris Presentation Attack Detection Based on Spatial and Frequency Feature Fusion

本仓库为[《基于空间域与频域特征融合的虹膜呈现攻击检测》](https://www.cjig.cn/thesisDetails#10.11834/jig.240783&lang=zh)论文中所提模型的代码仓库。

### 运行环境
本代码需要用到如下python库：
```
pytorch
timm
matplotlib
pandas
numpy
albumentations
sklearn
skimage
cv2
```

### 数据集
使用LivDet-Iris 2023比赛数据集作为测试集，该数据集可在[比赛官网](https://livdetiris23.github.io/)上申请

需要根据情况修改```dataset/data_config.py```中的相关路径

### 训练
运行
```
python train_simple.py
```

### 测试
从[这里](https://drive.google.com/drive/folders/1yDAmYAZjTHsBCiRrkRsTVnpLhkGTjdhn?usp=sharing)下载训练好的模型参数，并放入```ckpt```文件夹，

需要根据情将```train_config.py```中的```--ckpt```设置改为模型参数存放的路径。

之后运行
```
python test.py
```
更多设置可以查看```train_config.py```文件。
