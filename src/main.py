# -*- coding: utf-8 -*-

import pickle

import tensorflow as tf
from utils import create_model, get_logger
from model import Model
from loader import input_from_line
from train import FLAGS, load_config, train
import re
from zhon.hanzi import punctuation


# 导入模型
config = load_config(FLAGS.config_file)
logger = get_logger(FLAGS.log_file)
# limit GPU memory
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = False
with open(FLAGS.map_file, "rb") as f:
    tag_to_id, id_to_tag = pickle.load(f)
#创建tensorflow 会话（session）
sess = tf.Session(config=tf_config)
#初始化模型
model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

def reprocess(textList):

    resList = []
    for text in textList:
        line = re.sub("[{}]+".format(punctuation), "", text)#包含中文标点符号的识别结果，一般都是识别错误的
        if len(line)<len(text):
            continue
        if not line or len(line)<2:#文本长度较短的话，一般也是识别错误的
            continue
        line = line.replace('-',' ')
        resList.append(line)
    return resList

def outPutProcess(dataDict):
    # 为缓和数据少而导致的正确率不高的问题，这里对输出结果做过滤
    resDict = {}
    for key in dataDict:
        processedList = reprocess(dataDict[key])
        if not processedList:#如果结果为空，也过滤掉
            continue
        resDict[key] = processedList
    return resDict

def predict(text):
    text = text.replace(' ','-')
    #使用训练好的模型，对文本进行命名实体识别，结果会返回json格式的数据
    result = model.evaluate_line(sess, input_from_line(text, FLAGS.max_seq_len, tag_to_id), id_to_tag)
    return outPutProcess(result['data'])#输出的结果，再次进行过滤处理，辅助提高正确率


if __name__ == '__main__':
    string = '新型冠状病毒肺炎（Corona Virus Disease 2019，COVID-19），简称“新冠肺炎”，世界卫生组织命名为“2019冠状病毒病” [1-2]  ，是指2019新型冠状病毒感染导致的肺炎。'
    result = predict(string)
    print(result)