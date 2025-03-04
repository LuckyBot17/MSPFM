import argparse
import json
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
import torch
from src.acos_MultiPerspectives import BertForSequence, trainMulti, evaluateMulti, trainMulti_0
from tool_utils.feedback_utils.util import writeJson
from tool_utils.multi_utils import calculate_similarity, InputExample
from transformer_utils.models.bert.tokenization_bert import BertTokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
class InputExample_a(object):
    def __init__(self, guid, text_a, label_a,aspect_a):
        self.guid = guid
        self.text_a = text_a
        self.label_a = label_a
        self.aspect_a = aspect_a

def load_multi_examples(filename,dataname,K=-1):
    logger.info(f"Creating features from dataset file as {dataname}.")
    with open(os.path.join(filename,dataname+".json"),'r',encoding='utf-8') as load_file:
        datas = json.load(load_file)
    examples = []
    K = len(datas) if K == -1 else K
    for i in range(K):
        guid = f"{dataname},{i}"
        sentence = datas[i]['sentence']
        # sentence = datas[i]['raw_words']
        # quadruples = datas[i]['quadruples']
        # label = quadruples[0]['sentiment']
        label = datas[i]['label']
        aspect = datas[i]['aspect']
        # aspect = quadruples[0]['aspect']['term']
        examples.append(InputExample_a(guid=guid, text_a=sentence, label_a=label,aspect_a=aspect))
    logger.info(f"fload file {filename}/{dataname}.json,size{len(examples)}")
    return examples
def handle_MultiPerspectives():
    print("MultiPerspectives")
    task = args.task
    dataset = args.dataset
    train_idx = args.train_idx
    model_path = "./pre-trained_model/bert_uncased_L-12_H-768_A-12"
    bert_tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=True)
    bert_model = BertForSequence.from_pretrained(model_path,tokenizer=bert_tokenizer, max_seq_length=args.max_seq_length,
                                                 shot_num=args.shot_num,set_num=args.set_num,device=[0,1])
    bert_model.cuda()
    #bert_model = torch.nn.DataParallel(bert_model)
    # bert_model.to(args.device)
    train_examples = load_multi_examples(f"../data/{args.dataset}","train_1")
    test_examples = load_multi_examples(f"../data/{args.dataset}","test")
    if args.do_train:
        trainMulti(args,bert_model,train_examples)
        #trainMulti_0是多视角和预测反馈
        #trainMulti_0(args,bert_model,train_examples)
    # 将训练和测试样本计算得到的结果保存到MultiPerspectives_BERT文件夹下，用来下面进行样本相似度计算
    if not os.path.exists(f"./MultiPerspectives_BERT/{task}/{dataset}/"):
        os.makedirs(f"./MultiPerspectives_BERT/{task}/{dataset}/")
    if args.do_eval:
        # pytorch_model_path = f'../data/{dataset}/pytorch_model.bin'
        # checkpoint = torch.load(pytorch_model_path)
        # bert_model.load_state_dict(checkpoint)

        train_pt_text = evaluateMulti(args,bert_model,train_examples,out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/trains_{train_idx}_MultiAndSequential_text.pt",flag='text')
        test_pt_text = evaluateMulti(args,bert_model,test_examples,out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/tests_{train_idx}_MultiAndSequential_text.pt",flag='text')
        similarity_matrix_text = calculate_similarity(test_pt_text, train_pt_text)
        retrieve_toJson(args,similarity_matrix_text,perspective="text")

        train_pt_label = evaluateMulti(args,bert_model, train_examples,out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/trains_{train_idx}_MultiAndSequential_label.pt",flag='label')
        test_pt_label = evaluateMulti(args,bert_model, test_examples,out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/tests_{train_idx}_MultiAndSequential_label.pt", flag='label')
        similarity_matrix_label = calculate_similarity(test_pt_label, train_pt_label)
        retrieve_toJson(args, similarity_matrix_label,perspective="label")

        train_pt_pos = evaluateMulti(args,bert_model, train_examples, out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/trains_{train_idx}_MultiAndSequential_pos.pt",flag='pos')
        test_pt_pos = evaluateMulti(args,bert_model, test_examples, out_dir=f"./MultiPerspectives_BERT/{task}/{dataset}/tests_{train_idx}_MultiAndSequential_pos.pt", flag='pos')
        similarity_matrix_pos = calculate_similarity(test_pt_pos, train_pt_pos)
        retrieve_toJson(args, similarity_matrix_pos,perspective="pos")

        Averarge_Ranking3Perspectives(
            f"./ICL_examples/{args.task}/{args.dataset}/MultiAndSequential_{args.train_idx}_text.json",
            f"./ICL_examples/{args.task}/{args.dataset}/MultiAndSequential_{args.train_idx}_label.json",
            f"./ICL_examples/{args.task}/{args.dataset}/MultiAndSequential_{args.train_idx}_pos.json")

def Averarge_Ranking3Perspectives(text_path,label_path,pos_path):
    """
        根据从三个视角得到的相关度排名，最后得到平均排名，并将结果保存到average_ranks_per_input_sample.json
    """
    with open(text_path,'r') as text_path,\
            open(label_path,'r') as label_path,\
            open(pos_path,'r') as pos_path:
        data_text = json.load(text_path)
        data_label = json.load(label_path)
        data_pos = json.load(pos_path)

    result = {}
    # 遍历所有输入样本编号
    for input_sample in range(632):
        # 获取该输入样本的相关训练集样本排名列表
        label_ranks = data_label.get(str(input_sample), [])
        pos_ranks = data_pos.get(str(input_sample), [])
        text_ranks = data_text.get(str(input_sample), [])

        # 合并三个文件中该输入样本对应的训练集样本
        all_related_samples = set(label_ranks + pos_ranks + text_ranks)

        # 计算每个训练集样本的平均排名
        sample_avg_rank = {}
        for sample in all_related_samples:
            ranks = []
            for rank_list in [label_ranks, pos_ranks, text_ranks]:
                if sample in rank_list:
                    ranks.append(rank_list.index(sample) + 1)
                else:
                    ranks.append(len(rank_list) + 1)  # 若未出现，则赋予最大排名
            sample_avg_rank[sample] = sum(ranks) / len(ranks)

        # 将平均排名结果存入
        result[input_sample] = dict(sorted(sample_avg_rank.items(), key=lambda x: x[1]))  # 按平均排名排序
        # 保存到新的JSON文件
        output_file_path = f'./ICL_examples/ATSC/{args.dataset}/MultiAndSequential_average_ranks_per_input_sample.json'
        with open(output_file_path, 'w') as output_file:
            json.dump(result, output_file)

def retrieve_toJson(args,similarity_matrix,perspective):
    result_json = {}
    for i,similarity_scores in enumerate(similarity_matrix):
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices_list = sorted_indices.tolist()
        result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]
    #将样本相似度结果保存到ICL_examples文件夹下
    if not os.path.exists(f"./ICL_examples/{args.task}/{args.dataset}"):
        os.makedirs(f"./ICL_examples/{args.task}/{args.dataset}")
    writeJson(f"./ICL_examples/{args.task}/{args.dataset}/MultiAndSequential_{args.train_idx}_{perspective}.json", result_json, encoding="utf-8")


# def handle_Sequential():
#     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--icl_mode", type=str, default="random_each")
    parser.add_argument("--task", type=str, default="ATSC")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_idx", type=int, default=1, help="Index of training data")
    parser.add_argument("--shot_num", default=4)
    parser.add_argument("--set_num", default=8)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--learning_rate", default=3e-5)
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,choices=["WarmupLinearSchedule", "ConstantLRSchedule", "WarmupConstantSchedule", "WarmupCosineSchedule", "WarmupCosineWithHardRestartsSchedule"])
    parser.add_argument("--warmup_steps", default=0)
    parser.add_argument("--max_seq_length", default=128)
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_global_step", default=30, type=int)
    parser.add_argument("--retrieval_loss_weight",default=0.1)
    parser.add_argument("--do_train", type=bool,default=True)
    parser.add_argument("--do_eval", type=bool,default=False)
    args = parser.parse_args()
    args.icl_mode = "MultiPerspectives"
    args.dataset = "SemEvallap14"
    handlers = {
        "Multi":handle_MultiPerspectives()
    }
    handler_value = "Multi"
    handlers.get(handler_value,handle_MultiPerspectives())

