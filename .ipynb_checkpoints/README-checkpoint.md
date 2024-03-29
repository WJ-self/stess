# ST-ESS: **基于事件相机和帧相机融合的语义分割方法**

## 摘要

计算机视觉是人工智能的重要分支，语义分割是其关键技术，应用于图像理解和环境感知等领域。语义分割要求对图像中的每个像素进行分类，从而识别场景中的对象及其关系。传统的语义分割方法依赖于帧相机采集的图像数据，但在动态、快速变化的场景中受到光照、运动模糊等因素的影响，导致成像质量和性能下降。事件相机是一种新型的传感器，只在像素亮度变化时生成数据，具有微秒级的延迟、高动态范围和低功耗，能够捕捉快速移动的动态场景，适合于极端光照条件和实时场景。然而，由于事件数据的特性，传统的语义分割方法不能直接应用在事件相机数据之上，因此需要设计特有的算法。

为此，本研究围绕基于事件流的语义分割任务进行研究，提出一种融合事件相机和帧相机数据的语义分割方法。主要工作内容如下：

（1）针对有监督学习数据标注困难、数据量少的问题，提出了一种基于特征对齐的无监督领域适应语义分割方法，该方法的核心思想是通过最小化事件域和图像域之间的特征差异，实现跨域任务转移，从而将有标签的图像域数据的语义信息转移到无标签的事件域数据上，降低对标注数据的依赖。

（2）针对如何提取源域和目标域有用的特征表示问题，提出了一种结合脉冲神经网络（SNN）和Transformer的技术，SNN作为事件驱动的特征提取器，能够有效地捕捉这些事件之间的时间动态，而Transformer技术被引入作为图像语义特征提取的关键组件，具有高质量图像捕捉能力，这种混合架构既能够有效地处理事件数据的时空动态，又能够充分利用图像数据的高质量信息，从而提供更加全面和准确的环境感知能力。

（3）基于上述研究工作，本文在DDD17等多个公开的基于事件的数据集上对所提出的方法进行了广泛的实验评估，并与现有的方法进行了对比分析。实验结果表明，本研究提出的方法在mIoU和准确率上分别达到了55.98%和89.25%，优于基线方法约3.52%的mIoU和1.49%的准确率，且具有较好的鲁棒性和泛化性。

综上所述，本文的工作通过将事件相机与帧相机的优势结合起来，为自动驾驶等应用场景提供了更为准确和可靠的视觉感知方法。未来的工作将着重于进一步优化混合架构的性能，扩大其在不同场景和应用中的适用性。

## 安装

### 依赖

```bash
conda create -n <env_name>
```

```bash
pip install wheel
```

```bash
pip install -r requirements.txt
```

### 预训练的EVSNN模型

预训练的EVSNN模型可以在[这里](https://github.com/LinZhu111/EVSNN)下载，放在 `/evsnn/pretrained_models/`

## 数据集

### DSEC-Semantic

DSEC语义数据集可在[此处下载](https://dsec.ifi.uzh.ch/dsec-semantic/). 数据集应具有以下格式：

```
├── DSEC_Semantic
    │   ├── train
    │   │   ├── zurich_city_00_a
    │   │   │   ├── semantic
    │   │   │   │   ├── left
    │   │   │   │   │   ├── 11classes
    │   │   │   │   │   │   └──data
    │   │   │   │   │   │       ├── 000000.png
    │   │   │   │   │   │       └── ...
    │   │   │   │   │   └── 19classes
    │   │   │   │   │       └──data
    │   │   │   │   │           ├── 000000.png
    │   │   │   │   │           └── ...
    │   │   │   │   └── timestamps.txt
    │   │   │   └── events
    │   │   │       └── left
    │   │   │           ├── events.h5
    │   │   │           └── rectify_map.h5
    │   │   └── ...
    │   └── test
    │       ├── zurich_city_13_a
    │       │   └── ...
    │       └── ...
```

### DDD17

可以下载带有语义分割标签的预处理DDD17数据集[此处](https://download.ifi.uzh.ch/rpg/ESS/ddd17_seg.tar.gz).

### Cityscapes

Cityscapes数据集可在[此处下载](https://www.cityscapes-dataset.com/).

## 训练

可以在“config/settings_XXXX.yaml”中指定训练的设置。

下面的命令开始训练：

```
python train.py --settings_file config/settings_XXXX.yaml
```
如DDD17：
下面的命令开始训练：

```
python train.py --settings_file config/settings_DDD17.yaml
```

## 测试

为了测试预先训练的模型，在“config/settings_XXXX.yaml”中设置“load_pretrained_weights=True”，并在“pretrained_file”中指定预先训练的权重的路径。

## Pre-trained Weights

无监督领域适应的预训练权重下载如下：
