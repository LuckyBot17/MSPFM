import json
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import nltk
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from optimization import AdamW,Warmup
from tool_utils.feedback_utils.util import writeJson
from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from tool_utils.multi_utils import load_and_cache_examples, calculate_similarity

logger = logging.getLogger(__name__)

class BertForSequence(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_seq_length,shot_num,set_num,device,):
        super(BertForSequence, self).__init__(config)
        config.clue_num = 0
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ex_index = 0
        #self.bert = BertModel(config, output_attentions=False).to("cuda")
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        #尝试加入新部分
        self.gru_layer = 2
        self.shot_num = shot_num
        self.set_num =set_num
        self.device = device
        self.printN = 1
        self.decoder = torch.nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=self.gru_layer,
                                    bias=True,batch_first=True).to(dtype=torch.float16)

    # def multiPerspectives(self,indexs, examples, flag):
    #     all_input_ids, all_input_masks = [], []
    #     for index in indexs:
    #         if flag == 'text':
    #             text = examples[index].text_a
    #         if flag == 'text1':
    #             text = examples[index].text_a
    #
    #         if flag == 'label':
    #             aspect = examples[index].label_a[0]["aspect"]["term"][0]
    #             sentiment = examples[index].label_a[0]["sentiment"]
    #             sents = [f'the aspect " {aspect} " is {sentiment}']
    #             text = 'In this sentence , ' + ' , '.join(sents) + ' .'
    #
    #         if flag == 'pos':
    #             words = nltk.word_tokenize(examples[index].text_a)
    #             pos_tags = nltk.pos_tag(words)
    #             text = ' '.join([p for w, p in pos_tags])
    #
    #         text = text.lower()
    #         input_id, input_mask = self.convert_text_to_feature(text)
    #         #input_id = torch.tensor(input_id).unsqueeze(0).to(f"cuda:{self.device[0]}")
    #         input_id = torch.tensor(input_id).unsqueeze(0).cuda()
    #         #input_mask = torch.tensor(input_mask).unsqueeze(0).to(f"cuda:{self.device[0]}")
    #         input_mask = torch.tensor(input_mask).unsqueeze(0).cuda()
    #
    #         all_input_ids.append(input_id)
    #         all_input_masks.append(input_mask)
    #
    #     #input_ids = torch.cat(all_input_ids, dim=0).to(f"cuda:{self.device[0]}")
    #     input_ids = torch.cat(all_input_ids, dim=0)
    #     #input_masks = torch.cat(all_input_masks, dim=0).to(f"cuda:{self.device[0]}")
    #     input_masks = torch.cat(all_input_masks, dim=0)
    #     _, pool_output = self.bert(input_ids=input_ids, attention_mask=input_masks, output_all_encoded_layers=False)
    #     # del input_ids,input_masks
    #     # torch.cuda.empty_cache()
    #     hiddens = self.dropout(pool_output)
    #     return hiddens
    def multiPerspectives(self, indexs, examples, flag):
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
            input_id, input_mask = self.convert_text_to_feature(text)
            input_id = torch.tensor(input_id).unsqueeze(0).cuda()
            input_mask = torch.tensor(input_mask).unsqueeze(0).cuda()

            all_input_ids.append(input_id)
            all_input_masks.append(input_mask)

        input_ids = torch.cat(all_input_ids, dim=0)
        input_masks = torch.cat(all_input_masks, dim=0)
        _, pool_output = self.bert(input_ids=input_ids, attention_mask=input_masks, output_all_encoded_layers=False)
        hiddens = self.dropout(pool_output)
        return hiddens

    #新加
    #如何将不同角度的hidden_matrixs和initial_hidden输入到retrieval_demos中呢？
    def retrieval_demos(self, initial_index, initial_hidden, hidden_matrixs, set_num, mode):
        K = set_num
        T = self.shot_num
        hidden_matrixs = hidden_matrixs.to(f"cuda:{self.device[1]}",dtype=torch.float16)
        initial_hidden = initial_hidden.to(f"cuda:{self.device[1]}")
        example_num = hidden_matrixs.size(0)
        query_input = initial_hidden.unsqueeze(0).repeat(K, 1, 1)
        current_input = query_input.to(f"cuda:{self.device[1]}")
        # hx是GRU的初始隐藏状态，形状为[L, K, H]，其中L是GRU层数，H是隐藏状态的维度。
        hx = torch.zeros((self.gru_layer, K, self.config.hidden_size)).to(f"cuda:{self.device[1]}",dtype=torch.float16)

        all_sequences = []
        all_logits = []
        for t in range(T):
            current_input = current_input.to(dtype=torch.float16)
            self.decoder = self.decoder.to('cuda:1')
            output, hx = self.decoder(current_input, hx)
            # 计算当前解码器输出的样本与所有样本的隐藏状态的相似度。
            similarities = torch.matmul(input=output.squeeze(1), other=hidden_matrixs.t())
            # 通过 softmax 计算每个样本的选择概率。对每一行（每个查询），softmax 会产生一个大小为 example_num 的概率分布。
            probabilities = torch.nn.functional.softmax(similarities, dim=1)
            # 为了避免选择已经被选中的样本，需要创建一个掩码（mask）来屏蔽掉已选样本的概率。
            #mask = torch.ones_like(probabilities).bool().to(self.device)
            mask = torch.ones_like(probabilities).bool().to(f"cuda:{self.device[1]}")
            for idxs in all_sequences:
                mask[torch.arange(K), idxs] = False
            if mode == "train":
                mask[torch.arange(K), initial_index] = False
                # for i in range(4):
                #     mask[torch.arange(K), initial_index*(i+1)] = False


            probabilities = probabilities.masked_fill(mask == False, 0.0)
            # 在训练模式下，我们采用 混合概率分布：原始概率和均匀分布的加权平均。alpha 是调节混合度的超参数。
            if mode == "train":
                alpha = 0.1
                #uniform_distribution = torch.full(probabilities.shape, 1 / example_num).to(self.device)
                uniform_distribution = torch.full(probabilities.shape, 1 / example_num).to(f"cuda:{self.device[1]}")
                uniform_distribution = uniform_distribution.masked_fill(mask == False, 0.0)

                # 检查 probabilities 和 uniform_distribution 是否包含非法值
                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    print("Probabilities contain NaN or Inf before adjustment.")
                if torch.isnan(uniform_distribution).any() or torch.isinf(uniform_distribution).any():
                    print("Uniform distribution contains NaN or Inf before adjustment.")

                adjusted_probabilities = (1 - alpha) * probabilities + alpha * uniform_distribution

                # 修正 adjusted_probabilities 中的非法值
                adjusted_probabilities = torch.nan_to_num(adjusted_probabilities, nan=0.0, posinf=0.0, neginf=0.0)
                adjusted_probabilities = torch.clamp(adjusted_probabilities, min=0.0)

                next_indexs = torch.multinomial(adjusted_probabilities, 1).squeeze(1)
            if mode == "test":
                next_indexs = torch.argmax(probabilities, dim=1)
            # 将每次选择的样本索引 next_indexs 添加到 all_sequences 中。
            all_sequences.append(next_indexs)
            all_logits.append(torch.log(probabilities[torch.arange(K), next_indexs]))
            current_input = query_input


        # 将每轮选择的样本索引转化为元组，返回一个 K x T 的元组列表，表示每次检索的选择序列。
        tuples = [tuple(row.tolist()) for row in torch.stack(all_sequences, dim=1)]
        # 对每一行（每个查询）计算其日志概率之和，作为该查询的损失信号。
        tuples_logits = [torch.sum(row) for row in torch.stack(all_logits, dim=1)]
        # tuples 是选择的样本索引序列，tuples_logits 是每个检索序列的日志概率之和（表示其置信度）
        return tuples, tuples_logits

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

    def infonce_loss(self, T1, T2, temperature=0.5):
        assert T1.size() == T2.size()
        batch_size = T1.size(0)
        cosine_similarity = F.cosine_similarity(T1.unsqueeze(1), T2.unsqueeze(0), dim=2)
        labels = torch.arange(batch_size).to(T1.device)
        cosine_similarity /= temperature
        loss = F.cross_entropy(cosine_similarity, labels)
        return loss

def calculate_contrastive_losses(query_logits,positive_logits, negative_logits, temperature=0.5):
    """
        计算三对对数的损失: (neg, text), (neg, label),  (neg, pos).
        Args:
            tuple_logits_neg: List of Tensors containing negative tuple logits.
            tuples_logits_text: List of Tensors containing text tuple logits.
            tuples_logits_label: List of Tensors containing label tuple logits.
            tuples_logits_pos: List of Tensors containing pos tuple logits.
        Returns:
            dict: A dictionary with the computed losses.
    """
    # Combine all positive logits into a single tensor
    positive_logits_append = []
    for p in positive_logits:
        positive_logits_list = torch.stack(p,dim=0)
        positive_logits_append.append(positive_logits_list)
    positive_logits = torch.stack(positive_logits_append, dim=1)  # Shape: [K, num_positives]
    negative_logits = torch.stack(negative_logits, dim=0).unsqueeze(1)  # Shape: [K, num_negatives]
    query_logits = torch.stack(query_logits,dim=0)
    # Concatenate all logits (query, positives, negatives) for contrastive comparison
    all_logits = torch.cat([positive_logits, negative_logits], dim=1)  # Shape: [K, num_positives + num_negatives]

    # Compute similarity scores scaled by temperature
    scaled_logits =( all_logits / temperature).to('cuda:1')

    # Create ground truth labels for positive samples
    K = query_logits.size(0)
    labels = torch.arange(3).repeat(K // 3)  # Shape: [8]
    labels = torch.cat([labels, torch.zeros(K % 3, dtype=torch.long)]).to('cuda:1')  # Adjust for uneven batch sizes

    # Apply InfoNCE loss
    loss = F.cross_entropy(scaled_logits, labels)
    return loss.mean()


def get_AdamW(args,named_parameters,learning_rate,steps):
    # 提取模型中部分需优化参数，并以元组形式存储在列表中
    param_optimizer = [(k, v) for k, v in named_parameters if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]
    # 自定义AdamW优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # 学习率预热策略
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=steps)
    return  optimizer,scheduler

def Averarge_Ranking3Perspectives(args,text_path,label_path,pos_path):
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
        output_file_path = f'./acos_MultiPerspectives/{args.task}/{args.dataset}/average_ranks_per_input_sample.json'
        with open(output_file_path, 'w') as output_file:
            json.dump(result, output_file)

def retrieve_4shots(args,file,similarity_matrix):
    result_json = {}
    for i, similarity_scores in enumerate(similarity_matrix):
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices_list = sorted_indices.tolist()
        result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]

    with open(os.path.join(args.data_dir, file, file + "_similarity_results.json"), "w") as f:
       json.dump(result_json, f)
    #提取前四个最相似的样本索引
    top_shots = {}
    for key, value in result_json.items():
        # 假设 value 是一个包含排序索引的列表
        top_shots[int(key)] = value[:4]  # 取前四个索引
    return top_shots.items()

def retrieve_toJson(args,similarity_matrix,perspective):
    result_json = {}
    for i,similarity_scores in enumerate(similarity_matrix):
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices_list = sorted_indices.tolist()
        result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]
    #将样本相似度结果保存到ICL_examples文件夹下
    if not os.path.exists(f"./acos_MultiPerspectives/{args.task}/{args.dataset}"):
        os.makedirs(f"./acos_MultiPerspectives/{args.task}/{args.dataset}")
    writeJson(f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_{perspective}.json", result_json, encoding="utf-8")

def trainMulti_0(args,model,samples):
    """
    trainMulti_0是多视角
    """
    indexs = torch.arange(len(samples), dtype=torch.long)
    # 用tensordataset类型存储indexs
    INDEX = TensorDataset(indexs)
    # 一个epoch中，模型会看到的样本序号
    index_dataloader = DataLoader(INDEX, sampler=RandomSampler(INDEX), batch_size=args.batch_size)
    # 整个训练过程中的总步数，在处理完一步后，模型的参数会更新一次
    num_optimization_steps = len(index_dataloader) * args.num_train_epochs

    bert_optimizer, bert_scheduler = get_AdamW(args, model.bert.named_parameters(), learning_rate=1e-5,
                                               steps=args.num_train_epochs)
    logger.info("Training!!!")
    logger.info(f"Number of examples = {len(samples)}")
    logger.info(f"Number of optimization steps = {num_optimization_steps}")

    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for _, batch in enumerate(index_dataloader):
            one_embeds = model.multiPerspectives(batch[0].tolist(), samples, flag='text')
            two_embeds = model.multiPerspectives(batch[0].tolist(), samples, flag='text')
            loss = model.infonce_loss(one_embeds, two_embeds)

            label_embeds = model.multiPerspectives(batch[0].tolist(), samples, flag='label')
            loss2 = model.infonce_loss(one_embeds, label_embeds)

            pos_embeds = model.multiPerspectives(batch[0].tolist(), samples, flag='pos')
            loss3 = model.infonce_loss(one_embeds, pos_embeds)

            (loss + 0.1*loss2 + 0.1*loss3).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            bert_optimizer.step()
            bert_scheduler.step()
            model.zero_grad()

            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, Loss2:{:.5f}, Loss3:{:.5f}".format(epoch, global_step, num_optimization_steps, loss.item(), loss2.item(), loss3.item()))
            global_step += 1
def trainMulti(args, model, samples):
    """
        trainMulti是多视角和顺序检索
        """
    indexs = torch.arange(len(samples), dtype=torch.long)
    # 用tensordataset类型存储indexs
    INDEX = TensorDataset(indexs)
    # 一个epoch中，模型会看到的样本序号
    index_dataloader = DataLoader(INDEX, sampler=RandomSampler(INDEX), batch_size=args.batch_size)
    # 整个训练过程中的总步数，在处理完一步后，模型的参数会更新一次
    num_optimization_steps = len(index_dataloader) * args.num_train_epochs

    bert_optimizer,bert_scheduler = get_AdamW(args,model.bert.named_parameters(),learning_rate=1e-5,steps=args.num_train_epochs)
    decoder_optimizer,decoder_scheduler = get_AdamW(args,model.decoder.named_parameters(),learning_rate=1e-5,steps=num_optimization_steps)
    logger.info("Training!!!")
    logger.info(f"Number of examples = {len(samples)}")
    logger.info(f"Number of optimization steps = {num_optimization_steps}")

    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        #每个epoch的开始
        print(f"Straing epoch {epoch+1}/{args.num_train_epochs}")
        hidden_matrixs_total_text1,hidden_matrixs_total_text2,hidden_matrixs_total_label,hidden_matrixs_total_pos = [],[],[],[]
        for i, batch in enumerate(index_dataloader):

            hidden_matrixs_text_1 = model.multiPerspectives(batch[0].tolist(),samples,flag="text")
            hidden_matrixs_total_text1.append(hidden_matrixs_text_1)

            hidden_matrixs_text_2 = model.multiPerspectives(batch[0].tolist(), samples, flag="text")
            hidden_matrixs_total_text2.append(hidden_matrixs_text_2)
            loss = model.infonce_loss(hidden_matrixs_text_1, hidden_matrixs_text_2)

            hidden_matrixs_label = model.multiPerspectives(batch[0].tolist(), samples, flag="label")
            hidden_matrixs_total_label.append(hidden_matrixs_label)
            loss2 = model.infonce_loss(hidden_matrixs_text_1, hidden_matrixs_label)

            hidden_matrixs_pos = model.multiPerspectives(batch[0].tolist(), samples, flag="pos")
            hidden_matrixs_total_pos.append(hidden_matrixs_pos)
            loss3 = model.infonce_loss(hidden_matrixs_text_1, hidden_matrixs_pos)

            # 损失的反向传播与优化
            total_loss = loss + 0.1 * loss2 + 0.1 * loss3
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            bert_optimizer.step()
            bert_scheduler.step()
            model.zero_grad()
            print(
                "Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, Loss2:{:.5f}, Loss3:{:.5f}".format(epoch+1, global_step,
                                                                                              num_optimization_steps,
                                                                                              loss.item(), loss2.item(),
                                                                                              loss3.item()))

        # 在进行检索任务之前，将所有隐藏矩阵拼接成一个大矩阵
        hidden_matrixs_text_1_combined = torch.cat(hidden_matrixs_total_text1, dim=0).detach()
        hidden_matrixs_text_2_combined = torch.cat(hidden_matrixs_total_text2, dim=0).detach()
        hidden_matrixs_label_combined = torch.cat(hidden_matrixs_total_label, dim=0).detach()
        hidden_matrixs_pos_combined = torch.cat(hidden_matrixs_total_pos, dim=0).detach()
        finally_loss = []
        accmulation_steps = 4

        for _, batch in enumerate(index_dataloader):
            index_losses = []
            for index in batch[0]:
                index = index.item()
                # Step 1: 执行检索任务 (从多个视角检索)
                tuple_text_neg,tuple_logits_neg = model.retrieval_demos(initial_index=index,
                    initial_hidden=hidden_matrixs_text_1_combined[index],
                    hidden_matrixs=hidden_matrixs_text_1_combined,
                    set_num=model.set_num,mode="train")
                tuples_text, tuples_logits_text = model.retrieval_demos(initial_index=index,
                    initial_hidden=hidden_matrixs_text_2_combined[index],
                    hidden_matrixs=hidden_matrixs_text_2_combined,
                    set_num=model.set_num,mode="train")

                tuples_label, tuples_logits_label = model.retrieval_demos(initial_index=index,
                    initial_hidden=hidden_matrixs_label_combined[index],
                    hidden_matrixs=hidden_matrixs_label_combined,
                    set_num=model.set_num,mode="train")

                tuples_pos, tuples_logits_pos = model.retrieval_demos(initial_index=index,
                    initial_hidden=hidden_matrixs_pos_combined[index],
                    hidden_matrixs=hidden_matrixs_pos_combined,
                    set_num=model.set_num,mode="train")

                # Step 2: Compute contrastive loss
                contrastive_loss = calculate_contrastive_losses(
                    query_logits=tuples_logits_text,
                    positive_logits=[tuples_logits_text, tuples_logits_label, tuples_logits_pos],
                    negative_logits=tuple_logits_neg,
                    temperature=0.5
                )
                # Step 3: 反向传播和优化步骤：通过将不同角度的损失函数分配不同的权重，让不同角度检索的重要性不同
                index_losses.append(contrastive_loss)

            stacked_loss = torch.stack(index_losses)
            batch_loss = torch.mean(stacked_loss,dim=0)
            batch_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # if (i + 1) % accmulation_steps == 0:
            decoder_optimizer.step()
            decoder_scheduler.step()
            model.zero_grad()
            finally_loss.append(batch_loss)
            print(
                f"Epoch:{epoch+1}, Global Step:{global_step}/{num_optimization_steps}, Batch_Loss:{batch_loss.item():.5f}")
            if global_step % args.logging_global_step == 0:
                torch.save(model.state_dict(), os.path.join(f'../data/{args.dataset}', 'pytorch_model.bin'))
            global_step += 1

        finally_loss = torch.stack(finally_loss)
        finally_loss = torch.mean(finally_loss, dim=0)
        finally_loss.backward()

        # total_loss = loss + 0.1 * loss2 + 0.1 * loss3
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # bert_optimizer.step()
        # bert_scheduler.step()
        # model.zero_grad()

        print(f"Epoch{epoch+1}/{args.num_train_epochs} completed.Finally_loss:{finally_loss.item():.5f}")
        torch.cuda.empty_cache()
    print("Congratulation！！！！ Training completed")


def evaluateMulti(args, model, samples, out_dir, flag='text'):
    indexs = torch.arange(len(samples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    eval_dataloader = DataLoader(INDEX, sampler=SequentialSampler(INDEX), batch_size=args.batch_size)

    model.eval()
    out_reps = []
    for batch in eval_dataloader:
        with torch.no_grad():
            out = model.multiPerspectives(batch[0].tolist(), samples, flag=flag)
            out_reps.append(out)

    all_tensor = torch.cat(out_reps, dim=0)
    print(all_tensor.shape)
    torch.save(all_tensor, out_dir)
    return all_tensor

def main_MultiPerspectives():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--num_train_epochs",default=30,type=int)
    parser.add_argument("--learning_rate",default=3e-5)
    parser.add_argument("--schedule",default="WarmupLinearSchedule",type=str,choices=["WarmupLinearSchedule","ConstantLRSchedule","WarmupConstantSchedule","WarmupCosineSchedule","WarmupCosineWithHardRestartsSchedule"])
    parser.add_argument("--warmup_steps", default=0,)
    parser.add_argument("--data_dir",default="../data")
    parser.add_argument("--task",default="ATSC")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--max_seq_length",default=128)
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_global_step", default=200, type=int)
    parser.add_argument("--dataset",default="Clothing")
    parser.add_argument("--shot_num",default=4)
    parser.add_argument("--set_num",default=8)
    parser.add_argument("--do_train",type=bool)
    parser.add_argument("--do_eval",type=bool)
    args = parser.parse_args()
    args.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.do_train = True
    args.do_eval = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # file2model = {
    #     "lap14": "./pre-trained_model/bert_uncased_L-12_H-768_A-12",
    # }
    file2model = {
        "Books": "./pre-trained_model/bert_uncased_L-12_H-768_A-12",
    }

    for file, model_path in file2model.items():
        bert_tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=True)
        bert_model = BertForSequence.from_pretrained(model_path, tokenizer=bert_tokenizer, max_seq_length=args.max_seq_length,
                                                     shot_num=args.shot_num,set_num=args.set_num,device=[0,1])
        # # 将模型移到多块GPU
        # if torch.cuda.device_count() > 1:
        #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
        #bert_model = torch.nn.DataParallel(bert_model)  # 使用 DataParallel 将模型并行到多个 GPU 上
        train_examples = load_and_cache_examples(args, file, dataname="train_1")
        if args.do_train:
            trainMulti(args,bert_model,train_examples)
        # 先使用前100个样本输入到模型中训练模型，让模型从多个视角关注样本，然后用模型进行evaluate，得到.pt文件，文件中包含800个test句子的每个的相似度得分，
        # 然后将三个角度得到的.pt文件得到分数之和进行最终排名，最后检索出前k个示例作为上下文的示例
        if args.do_eval:
            test_examples = load_and_cache_examples(args, file, dataname="test")

            logger.info("load checkpoint...%s",os.path.join(args.data_dir,'pytorch.bin'))

            trains_pt_text = evaluateMulti(args, bert_model, train_examples, out_dir=os.path.join(args.data_dir, file, "train-text.pt"),flag='text')
            tests_pt_text = evaluateMulti(args, bert_model, test_examples, out_dir=os.path.join(args.data_dir, file, "test-text.pt"),flag='text')
            similarity_matrix_text = calculate_similarity(tests_pt_text, trains_pt_text)
            retrieve_toJson(args, similarity_matrix_text, perspective="text")
            logging.info("text-perspectives_pt has done!")

            trains_pt_label = evaluateMulti(args, bert_model, train_examples, out_dir=os.path.join(args.data_dir, file, "train-label.pt"),flag='label')
            tests_pt_label = evaluateMulti(args, bert_model, test_examples, out_dir=os.path.join(args.data_dir, file, "test-label.pt"),flag='label')
            similarity_matrix_label = calculate_similarity(tests_pt_label, trains_pt_label)
            retrieve_toJson(args, similarity_matrix_label, perspective="label")
            logging.info("label-perspectives_pt has done!")

            trains_pt_pos = evaluateMulti(args, bert_model, train_examples, out_dir=os.path.join(args.data_dir, file, "train-pos.pt"), flag='pos')
            tests_pt_pos = evaluateMulti(args, bert_model, test_examples, out_dir=os.path.join(args.data_dir, file, "test-pos.pt"), flag='pos')
            similarity_matrix_pos = calculate_similarity(tests_pt_pos, trains_pt_pos)
            retrieve_toJson(args, similarity_matrix_pos, perspective="pos")
            logging.info("pos-perspectives_pt has done!")

            Averarge_Ranking3Perspectives(args,f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_text.json",
                                          f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_label.json",
                                          f"./acos_MultiPerspectives/{args.task}/{args.dataset}/MultiPerspectives_pos.json")
            # shots = retrieve_4shots(args,file,similarity_matrix)
            # # 打印找到的样例排在前四个的索引序号
            # for test_idx, shots in shots:
            #     print(f"Test sample {test_idx} most similar shots: {shots}")

if __name__ == '__main__':
    main_MultiPerspectives()