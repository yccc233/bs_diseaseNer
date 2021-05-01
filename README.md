
# 新冠相关的医学实体命名抽取

## 数据已经训练完成，训练时间约半小时——MacOS

## 环境

主要需要3个包，pip安装方法直接给出：

```shell
pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl
pip3 install zhon -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意，tensorflow的版本最好是1.14版本的**


**开发环境python支持3.6。不符合此条件的在main时会出错**


## 使用方法

1. tran.py运行可训练
2. main.py运行即可，string字符串可自定义，返回json串