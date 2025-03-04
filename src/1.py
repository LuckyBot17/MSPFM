import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

#train_df = pd.read_json("../data/lap14/train_1.json")
# df_expanded = train_df.explode("quadruples")
# quadruples_df = pd.json_normalize(df_expanded["quadruples"])
# term = quadruples_df["aspect.term"]
# print(term)
#stat_dict = torch.load(f"{model_path}/pytorch_model-00001-of-00002.bin",map_location="cpu")
#print(stat_dict.keys())
# import json
#
# # 读取上传的 JSON 文件
# input_path = "../data/lap14/train_1.json"
# output_path = "train_1.json"
#
# with open(input_path, "r") as file:
#     data = json.load(file)
#
# # 获取数据的键
# raw_words = data["raw_words"]
# task = data["task"]
# quadruples = data["quadruples"]
# sentiment = data["sentiment"]
# prediction = data["prediction"]
#
# # 转换为指定格式
# transformed_data = []
# for key in range(len(input_path)):
#     transformed_data.append({
#         "raw_words": raw_words[key],
#         "task": task[key],
#         "quadruples": quadruples[key],
#         "sentiment": sentiment[key],
#         "prediction": prediction[key]
#     })
#
# # 保存到新的 JSON 文件
# with open(output_path, "w") as output_file:
#     json.dump(transformed_data, output_file, indent=4)
#
# print(f"数据已成功保存到 {output_path}")
import json
from sklearn.metrics import f1_score, precision_score, recall_score
# 读取 JSON 文件
with open('../result/meta-llama2-7b-chat/ATSC/SemEvallap14/train1_MultiPerspectives_Multi_4_icl.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取真实标签 (Label) 和预测标签 (outputs)
true_labels = [item['label'] for item in data]  # 假设 JSON 中的每一项包含 'Label'
predicted_labels = [item['outputs'] for item in data]  # 假设 JSON 中的每一项包含 'outputs'

# 计算 F1 Score
f1 = f1_score(true_labels, predicted_labels, average='weighted')  # weighted: 加权F1, 可以根据情况选择其他方式
precision = precision_score(true_labels, predicted_labels, average='weighted')  # 加权精确度
recall = recall_score(true_labels, predicted_labels, average='weighted')  # 加权召回率

print(f"F1 Score: {f1}")#F1 Score: F1 Score: 0.7594900426801717
print(f"Precision: {precision}")#Precision: 0.7570978466201269
print(f"Recall: {recall}")#Recall: 0.7804459691252144

# def trainMulti_0(args,model,samples):
#     indexs = torch.arange(len(samples), dtype=torch.long)
#     # 用tensordataset类型存储indexs
#     INDEX = TensorDataset(indexs)
#     # 一个epoch中，模型会看到的样本序号
#     index_dataloader = DataLoader(INDEX, sampler=RandomSampler(INDEX), batch_size=args.batch_size)
#     # 整个训练过程中的总步数，在处理完一步后，模型的参数会更新一次
#     num_optimization_steps = len(index_dataloader) * args.num_train_epochs
#
#     bert_optimizer, bert_scheduler = get_AdamW(args, model.named_parameters(), learning_rate=1e-5,
#                                                steps=args.num_train_epochs)
#     decoder_optimizer, decoder_scheduler = get_AdamW(args, model.decoder.named_parameters(), learning_rate=1e-5,
#                                                      steps=num_optimization_steps)
#     logger.info("Training!!!")
#     logger.info(f"Number of examples = {len(samples)}")
#     logger.info(f"Number of optimization steps = {num_optimization_steps}")
#
#     model.zero_grad()
#     model.train()
#     global_step = 0
#     for epoch in range(int(args.num_train_epochs)):
#         # 每个epoch的开始
#         print(f"Straing epoch {epoch + 1}/{args.num_train_epochs}")
#         finally_loss = []
#         accmulation_steps = 4
#         shengyu = len(samples)%args.batch_size
#         # Step 1: 获取多视角表示
#         print("Computing one_embeds...")
#         one_embeds = model.module.multiPerspectives(range(len(samples)), samples, flag='text')
#         print("Computing two_embeds...")
#         two_embeds = model.module.multiPerspectives(range(len(samples)), samples, flag='text1')
#         print("Computing label_embeds...")
#         label_embeds = model.module.multiPerspectives(range(len(samples)), samples, flag='label')
#         print("Computing pos_embeds...")
#         pos_embeds = model.module.multiPerspectives(range(len(samples)), samples, flag='pos')
#
#         # Step 2: 计算表征间的对比损失
#         loss_text_text = model.module.infonce_loss(one_embeds, two_embeds)
#         loss_text_label = model.module.infonce_loss(one_embeds, label_embeds)
#         loss_text_pos = model.module.infonce_loss(one_embeds, pos_embeds)
#         total_rep_loss = loss_text_label + loss_text_pos + loss_text_text
#         total_rep_loss = total_rep_loss.to("cuda:1")
#
#         for i, batch in enumerate(index_dataloader):
#             batch_indexs = batch[0].tolist()
#
#             # Step 1: 获取多视角表示
#             print("Computing one_embeds...")
#             one_embeds = model.module.multiPerspectives(batch[0].tolist(), samples, flag='text')
#             print("Computing two_embeds...")
#             two_embeds = model.module.multiPerspectives(batch[0].tolist(), samples, flag='text1')
#             print("Computing label_embeds...")
#             label_embeds = model.module.multiPerspectives(batch[0].tolist(), samples, flag='label')
#             print("Computing pos_embeds...")
#             pos_embeds = model.module.multiPerspectives(batch[0].tolist(), samples, flag='pos')
#
#             # Step 2: 计算表征间的对比损失
#             loss_text_text = model.module.infonce_loss(one_embeds, two_embeds)
#             loss_text_label = model.module.infonce_loss(one_embeds, label_embeds)
#             loss_text_pos = model.module.infonce_loss(one_embeds, pos_embeds)
#             total_rep_loss = loss_text_label + loss_text_pos + loss_text_text
#             total_rep_loss = total_rep_loss.to("cuda:1")
#
#             Step 3: 使用多视角表示进行样本检索
#
#             _, retrieval_logits = model.module.retrieval_demos(
#                 initial_index=torch.tensor(batch_indexs).to(args.device),
#                 initial_hidden=one_embeds,
#                 hidden_matrixs=hidden_matrix,
#                 set_num=args.set_num,
#                 mode="train"
#             )
#             if batch_indexs.index(batch_indexs[i%args.batch_size]) <shengyu :
#             initial_batch_index = batch_indexs.index(batch_indexs[i%args.batch_size])
#             initial_batch_index = [i for i in range(len(batch))]
#
#
#             _, retrieval_logits = model.module.retrieval_demos(initial_index=initial_batch_index,initial_hidden=hidden_matrix[initial_batch_index],hidden_matrixs=hidden_matrix,set_num=args.set_num,mode="train" )
#             retrieval_loss = -torch.mean(torch.stack(retrieval_logits))
#
#             # Step 4: 综合损失并反向传播
#             batch_loss = total_rep_loss + args.retrieval_loss_weight * retrieval_loss
#             batch_loss.backward()
#             finally_loss.append(batch_loss)
#
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#             if (i + 1) % accmulation_steps == 0:
#                 decoder_optimizer.step()
#                 decoder_scheduler.step()
#             model.zero_grad()
#
#             # Logging
#             if global_step % args.logging_global_step == 0:
#                 logger.info(
#                     "Epoch:{}, Global Step:{}/{}, Batch_Loss:{:.5f}, Retrieval_Loss:{:.5f}".format(
#                         epoch, global_step, num_optimization_steps, total_rep_loss.item(), retrieval_loss.item()
#                     )
#                 )
#                 torch.save(model.state_dict(),os.path.join(f'../data/{args.dataset}','pytorch_model.bin'))
#             global_step += 1
#
#         finally_loss = torch.stack(finally_loss)
#         finally_loss = torch.mean(finally_loss,dim=0)
#         finally_loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
#         bert_optimizer.step()
#         bert_scheduler.step()
#         model.module.zero_grad()
#         print(f"Epoch{epoch + 1}/{args.num_train_epochs} completed.Finally_loss:{finally_loss.item():.5f}")
#         torch.cuda.empty_cache()
#     print("Congratulation!!! Training completed")
# import json
#
# # 读取原始JSON文件
# def load_json(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data
#
# # 转换数据格式
# def transform_data(data):
#     transformed_data = []
#     for i in range(len(data['id'])):
#         entry = {
#             'id': data['id'][str(i)],
#             'input_all_tokens': data['input_all_tokens'][str(i)],
#             'label': data['label'][str(i)]
#         }
#         transformed_data.append(entry)
#     return transformed_data
#
# # 保存转换后的数据到新的JSON文件
# def save_transformed_data(data, output_file_path):
#     with open(output_file_path, 'w', encoding='utf-8') as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)
#
# # 主函数
# def main():
#     input_file_path = '../result/meta-llama2-7b-chat/ATSC/lap14/train_1_pre.json'  # 替换为你的输入文件路径
#     output_file_path = '../result/meta-llama2-7b-chat/ATSC/lap14/train_1_pre.json'  # 替换为你想要的输出文件路径
#
#     data = load_json(input_file_path)
#     transformed_data = transform_data(data)
#     save_transformed_data(transformed_data, output_file_path)
#     print(f"Transformed data saved to {output_file_path}")
#
# if __name__ == "__main__":
#     main()
# huggingface-cli download --resume-download NousResearch/Llama-2-13b-chat-hf --local-dir Llama-2-13b-chat-hf --local-dir-use-symlinks False --include "pytorch_model-00002-of-00003.bin"
#
# huggingface-cli download --resume-download NousResearch/Llama-2-70b-chat-hf --local-dir Llama-2-70b-chat-hf --local-dir-use-symlinks False --include "pytorch_model*.bin"

