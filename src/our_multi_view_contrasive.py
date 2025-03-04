# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

from src.caculate import InputExample_a

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import random
import sys

from src.acos_MultiPerspectives import retrieve_toJson, Averarge_Ranking3Perspectives

sys.path.append('../multi-perspectives_example_retrieval/')
sys.path.append("../")
import json
import nltk
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from optimization import AdamW, WarmupLinearSchedule, Warmup
from tool_utils.multi_utils import load_and_cache_examples, calculate_similarity, InputExample

logger = logging.getLogger(__name__)
torch.set_num_threads(12)


class BertForSequence1(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_seq_length):
        super(BertForSequence1, self).__init__(config)
        config.clue_num = 0
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ex_index = 0
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def convert_text_to_feature(self, text):
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token_id = 0

        tokens = []
        for tok in text.split():
            tokens.extend(self.tokenizer.wordpiece_tokenizer.tokenize(tok))
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]

        tokens = [cls_token] + tokens + [sep_token]        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        assert len(input_ids) == len(input_mask) == self.max_seq_length
        
        if self.ex_index < 1:
            logger.info("tokens: %s" % text)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            self.ex_index += 1
        return input_ids, input_mask

    def multiPerspectives1(self, indexs, examples, flag):
        all_input_ids, all_input_masks = [], []
        for index in indexs:
            if flag == 'text':
                text = examples[index].text_a

            if flag == 'label':
                # aspect = examples[index].label_a[0]["aspect"]["term"][0]
                # sentiment = examples[index].label_a[0]["sentiment"]
                # sents = [f'the aspect " {aspect} " is {sentiment}']
                aspect = examples[index].aspect_a
                sentiment = examples[index].label_a
                sents = [f'the aspect "{aspect}" is {sentiment}']
                text = 'In this sentence , ' + ' , '.join(sents) + ' .'
            
            if flag == 'pos':
                words = nltk.word_tokenize(examples[index].text_a)
                pos_tags = nltk.pos_tag(words)
                text = ' '.join([p for w, p in pos_tags])

            text = text.lower()
            input_id, input_mask = self. convert_text_to_feature(text)
            input_id = torch.tensor(input_id).unsqueeze(0).cuda()
            input_mask = torch.tensor(input_mask).unsqueeze(0).cuda()

            all_input_ids.append(input_id)
            all_input_masks.append(input_mask)

        input_ids = torch.cat(all_input_ids, dim=0)
        input_masks = torch.cat(all_input_masks, dim=0)
        _, pool_output = self.bert(input_ids=input_ids, attention_mask=input_masks, output_all_encoded_layers=False)
        hiddens = self.dropout(pool_output)
        return hiddens

    def infonce_loss(self, T1, T2, temperature=0.5):
        assert T1.size() == T2.size()
        batch_size = T1.size(0)
        cosine_similarity = F.cosine_similarity(T1.unsqueeze(1), T2.unsqueeze(0), dim=2)
        labels = torch.arange(batch_size).to(T1.device)
        cosine_similarity /= temperature
        loss = F.cross_entropy(cosine_similarity, labels)
        return loss


def train(args, model, samples):
    indexs = torch.arange(len(samples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    index_dataloader = DataLoader(INDEX,sampler=RandomSampler(INDEX), batch_size=args.batch_size)
    num_optimization_steps = len(index_dataloader) * args.num_train_epochs

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(samples))
    logger.info("Total optimization steps = %d", num_optimization_steps)

    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for _, batch in enumerate(index_dataloader):
            one_embeds = model.multiPerspectives1(batch[0].tolist(), samples, flag='text')
            two_embeds = model.multiPerspectives1(batch[0].tolist(), samples, flag='text')
            loss = model.infonce_loss(one_embeds, two_embeds)

            label_embeds = model.multiPerspectives1(batch[0].tolist(), samples, flag='label')
            loss2 = model.infonce_loss(one_embeds, label_embeds)

            pos_embeds = model.multiPerspectives1(batch[0].tolist(), samples, flag='pos')
            loss3 = model.infonce_loss(one_embeds, pos_embeds)

            (loss + 0.1*loss2 + 0.1*loss3).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, Loss2:{:.5f}, Loss3:{:.5f}".format(epoch, global_step, num_optimization_steps, loss.item(), loss2.item(), loss3.item()))
            global_step += 1


def evaluate(args, model, samples, out_dir, flag='text'):
    indexs = torch.arange(len(samples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    eval_dataloader = DataLoader(INDEX, sampler=SequentialSampler(INDEX), batch_size=args.batch_size)
    
    model.eval()
    out_reps = []
    for batch in eval_dataloader:
        with torch.no_grad():
            out = model.multiPerspectives1(batch[0].tolist(), samples, flag=flag)
            out_reps.append(out)

    all_tensor = torch.cat(out_reps, dim=0)
    print(all_tensor.shape)
    torch.save(all_tensor, out_dir)
    return all_tensor

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
        sentence = datas[i]['sentence']
        # sentence = datas[i]['raw_words']
        # quadruples = datas[i]['quadruples']
        # label = quadruples[0]['sentiment']
        label = datas[i]['label']
        aspect = datas[i]['aspect']
        # aspect = quadruples[0]['aspect']['term']
        examples.append(InputExample_a(guid=guid, text_a=sentence, label_a=label,aspect_a=aspect))
    logger.info(
        "load file :{}, size :{}".format(os.path.join(args.data_dir, file, dataname + ".json"), len(examples)))
    return examples

def retrieve_4shots(args,file,similarity_matrix):
    result_json = {}
    for i, similarity_scores in enumerate(similarity_matrix):
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices_list = sorted_indices.tolist()
        result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]

    # with open(os.path.join(args.data_dir, file, file + "_similarity_results.json"), "w") as f:
    #     json.dump(result_json, f)
    # 提取前四个最相似的样本索引
    top_shots = {}
    for key, value in result_json.items():
        # 假设 value 是一个包含排序索引的列表
        top_shots[int(key)] = value[:4]  # 取前四个索引
    return top_shots.items()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data/', type=str)

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=25, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
                        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    parser.add_argument("--logging_global_step", default=50, type=int)
    parser.add_argument("--task", default="ATSC")
    parser.add_argument("--dataset", default="SemEvallap14")
    parser.add_argument("--shot_num", default=4)
    parser.add_argument("--set_num", default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    file2model = {
            f"{args.dataset}":"./pre-trained_model/bert_uncased_L-12_H-768_A-12",
        }

    for file, model_path in file2model.items():
        bert_tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=True)
        bert_model = BertForSequence1.from_pretrained(model_path, tokenizer=bert_tokenizer, max_seq_length=args.max_seq_length)
        bert_model.cuda()

        train_examples = load_and_cache_examples(args, file, dataname="train", K=99)
        train(args, bert_model, train_examples)
        # 先使用前100个样本输入到模型中训练模型，让模型从多个视角关注样本，然后用模型进行evaluate，得到.pt文件，文件中包含800个test句子的每个的相似度得分，
        # 然后将三个角度得到的.pt文件得到分数之和进行最终排名，最后检索出前k个示例作为上下文的示例
        test_examples = load_and_cache_examples(args, file, dataname="test")

        trains_pt_text = evaluate(args, bert_model, train_examples,
                                       out_dir=os.path.join(args.data_dir, file, "train-text.pt"), flag='text')
        tests_pt_text = evaluate(args, bert_model, test_examples,
                                      out_dir=os.path.join(args.data_dir, file, "test-text.pt"), flag='text')
        similarity_matrix_text = calculate_similarity(tests_pt_text, trains_pt_text)
        retrieve_toJson(args, similarity_matrix_text, perspective="text")
        logging.info("text-perspectives_pt has done!")

        trains_pt_label = evaluate(args, bert_model, train_examples,
                                        out_dir=os.path.join(args.data_dir, file, "train-label.pt"), flag='label')
        tests_pt_label = evaluate(args, bert_model, test_examples,
                                       out_dir=os.path.join(args.data_dir, file, "test-label.pt"), flag='label')
        similarity_matrix_label = calculate_similarity(tests_pt_label, trains_pt_label)
        retrieve_toJson(args, similarity_matrix_label, perspective="label")
        logging.info("label-perspectives_pt has done!")

        trains_pt_pos = evaluate(args, bert_model, train_examples,
                                      out_dir=os.path.join(args.data_dir, file, "train-pos.pt"), flag='pos')
        tests_pt_pos = evaluate(args, bert_model, test_examples,
                                     out_dir=os.path.join(args.data_dir, file, "test-pos.pt"), flag='pos')
        similarity_matrix_pos = calculate_similarity(tests_pt_pos, trains_pt_pos)
        retrieve_toJson(args, similarity_matrix_pos, perspective="pos")
        logging.info("pos-perspectives_pt has done!")

        Averarge_Ranking3Perspectives(args,
                                      f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_text.json",
                                      f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_label.json",
                                      f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_pos.json")



if __name__ == "__main__":
    main()


