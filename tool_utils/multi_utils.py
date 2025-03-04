import json
import logging
import os.path

import torch

logger = logging.getLogger(__name__)
class InputExample(object):
    def __init__(self, guid, text_a, label_a):
        self.guid = guid
        self.text_a = text_a
        self.label_a = label_a

def load_and_cache_examples(args,file,dataname,K=-1):
    """
    Multiperspectives
    加载样本
    样本中有raw_words和quadruples，quadruples中包含四元组的各个方面
    """
    logger.info(f"Creating features from dataset file as {args.data_dir}---{file}---{dataname}")
    with open(os.path.join(args.data_dir,file,dataname+".json"),'r',encoding='utf-8') as load_file:
        datas = json.load(load_file)

    examples = []
    K = len(datas) if K == -1 else K
    for i in range(K):
        guid = f"{dataname},{i}"
        sentence = datas[i]['raw_words']
        quadruples = datas[i]['quadruples']
        if i<2:
            print(sentence,quadruples)
        examples.append(InputExample(guid=guid, text_a=sentence, label_a=quadruples))
    logger.info(
        "load file :{}, size :{}".format(os.path.join(args.data_dir, file, dataname + ".json"), len(examples)))
    return examples

def calculate_similarity(tensor_A, tensor_B):
    """
计算两组文本表示之间的相似度矩阵
    :param tensor_A: 二维矩阵
    :param tensor_B: 二维矩阵
    :return:
    """
    norm_A = torch.nn.functional.normalize(tensor_A, p=2, dim=1)
    norm_B = torch.nn.functional.normalize(tensor_B, p=2, dim=1)
    similarity_matrix = torch.mm(norm_A, norm_B.t())
    return similarity_matrix