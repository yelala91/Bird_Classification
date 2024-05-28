# 微调ResNet18适应鸟类分类任务

本次作业主要在 CUB_200_2011 微调预训练好的 ResNet18 模型，从而适应细粒度鸟类分类的任务.

## 训练

首先在终端运行如下代码对数据集进行转换
```
python private/data_convert.py
```

### 微调预训练的模型
把 ```demo.py``` 中的变量 ```pretrained``` 修改为 ```True```, 然后在终端运行
```
python demo.py
```
即可开始训练.

### 从零开始训练
把前述的变量 ```pretrained``` 修改为 ```True```即可.

训练过程中生成的所有日志和权重都会保存在 ```saves``` 目录下.

## 测试

先下载训练好的权重, 下载地址在报告的附录中. 然后在终端运行如下命令
```
python test.py ${下载好的权重地址}$
```
即可测试模型在测试集上的准确率.