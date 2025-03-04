import json
import os

import pandas as pd

pool_size = 100 # candidate example pool
# config
train_data = 'SemEvallap14'
train_idx = 1  # 1, 2, 3
seed = 75  # different seeds for different train_idx

# train_df = pd.read_json(f"./{data}/train.json")
# df_expanded = train_df.explode("quadruples")
# quadruples_df = pd.json_normalize(df_expanded["quadruples"])
# sentiment = quadruples_df["sentiment"]
# df_expanded["sentiment"] = quadruples_df["sentiment"].values

# examples_per_label = pool_size // (len(sentiment.unique())-1) #33
# # 按 sentiment 排序
# df_expanded_sorted = df_expanded.groupby("polarity",group_keys=False).apply(lambda x: x.sample(examples_per_label, random_state=seed))
# with open(f"./{data}/train_{train_idx}.json","w",encoding='utf-8') as save_file:
#     json.dump(df_expanded_sorted.to_dict(orient='records'),save_file,indent=4)

import json
import random

# 加载原始数据
with open(f'./{train_data}/train.json', 'r') as f:
    data = json.load(f)

# 按照label类型将数据分组
label_groups = {}
for entry in data:
    label = entry['label']
    if label not in label_groups:
        label_groups[label] = []
    label_groups[label].append(entry)

# 用于存储选择的句子，确保没有重复
chosen_sentences = set()
random_data = []

# 为每个label类型随机选择33条数据
for label, entries in label_groups.items():
    # 确保每个组有足够的数据
    if len(entries) >= 33:
        # 过滤掉已选择的句子，避免重复
        unique_entries = [entry for entry in entries if entry['sentence'] not in chosen_sentences]

        if len(unique_entries) >= 33:
            # 随机选择33条不重复的句子
            random_entries = random.sample(unique_entries, 33)

            # 将选择的句子加入已选择句子集合
            for entry in random_entries:
                chosen_sentences.add(entry['sentence'])

            # 将选中的数据加入最终结果
            random_data.extend(random_entries)

# 生成新JSON文件
with open(f'./{train_data}/train_{train_idx}.json', 'w') as f:
    json.dump(random_data, f, indent=4)

print("新数据已保存到 'train_1.json' 文件中")





