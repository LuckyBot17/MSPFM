import argparse
import json
import os.path

import pandas as pd
import sys
sys.path.append("../")
from tool_utils.feedback_utils.util import *
from tool_utils.feedback_utils.data_utils import *
from tool_utils.feedback_utils.generate import LLMgenerate

from torch.utils.data import DataLoader
from ICL.ICL_templates import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 获得LLM的先验预测
def format_eval_output(rows):
    idx, tokens, labels, outputs = zip(*rows)
    idx = np.vstack(idx)
    tokens = np.vstack(tokens)
    labels = np.vstack(labels)
    outputs = np.vstack(outputs)
    results_df = pd.DataFrame()
    results_df["id"] = idx.reshape(-1).tolist()
    results_df["input_all_tokens"] = tokens.reshape(-1).tolist()
    results_df["label"] = labels.reshape(-1).tolist()
    results_df["outputs"] = outputs.reshape(-1).tolist()
    return results_df

def llm_pre_main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model",type=str,default="meta-llama2-7b-chat",help="LLM directory / name")
    parser.add_argument("--dataset",type=str,help="Dataset to test on",)
    parser.add_argument("--train_examples", type=int, default=99, help="Number of training data")
    parser.add_argument("--task",type=str, choices=["ATSC", "SC", "EMO"],help="Define task name")
    parser.add_argument("--train_idx", type=int, default=1, help="Index of training data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed" )
    parser.add_argument("--run_pre", action="store_true", default=False, help="Run prior prediction generation")
    parser.add_argument("--run_icl", action="store_true", default=False, help="Run ICL baseline")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Inference")
    parser.add_argument("--load_bit", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="train", help="running mode")
    parser.add_argument("--max_len", type=int, default=500)
    parser.add_argument( "--write_to_train", action="store_true", default=True)
    args = parser.parse_args()
    args.run_pre = True
    args.task = "ATSC"
    args.dataset = "SemEvallap14"
    args.train_idx = 1
    args.mode = "train"
    args.write_to_train = True
    # seed & log
    seed_everything(seed=args.seed)
    # basic info
    print(f"LLM: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Running-Mode: {args.mode}")
    print(f"Loading Tokenizer")
    print(f"Number of Training examples: {args.train_examples}")
    tokenizer = Tokenizer4LLM(args.max_len, args.model, llm=True)

    dataset_name = args.dataset
    # load dataset
    train_df = pd.read_json(f"../data/{dataset_name}/train_{args.train_idx}.json")
    test_df = pd.read_json(f"../data/{dataset_name}/test.json")
    if args.run_pre:
        if args.mode == "train":
            test_df = train_df
        prompt = icl_instruction_prompt[dataset_name]
        TestDataset = ClassificationDataset(test_df, args, prompt, get_input_template, tokenizer=tokenizer)

        test_loader = DataLoader(TestDataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                 drop_last=False)

        print("Loading LLM")
        all_labels = label_space[dataset_name]
        LLM = LLMgenerate(args, tokenizer, all_labels=all_labels)
        # Results
        rows = []
        print("Start Generation with the LLM")

        with torch.no_grad():
            for ii, d in enumerate(test_loader):
                print(ii)
                if ii <= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)
                print(output)
                rows.extend(zip(d['index'],d['input_tokens'],d["labels"],output))
        result_df = format_eval_output(rows)
        if not os.path.exists(f"../result/{args.model}/{args.task}/{args.dataset}") :
            os.makedirs(f"../result/{args.model}/{args.task}/{args.dataset}")
        # 将结果保存到result文件夹下
        result_df.to_json(f"../result/{args.model}/{args.task}/{args.dataset}/{args.mode}_{args.train_idx}_pre.json")
        if args.write_to_train:
            #输出各个outputs列中，各个结果出现的次数
            print(result_df["outputs"].value_counts())
            #将train_1_pre.json中的outputs写入到train_1.json中的prediction中
            train_df = pd.read_json(f"../data/{args.dataset}/train_{args.train_idx}.json")
            for i in range(len(train_df)):
                #assert train_df.iloc[i]["sentiment"] == result_df.iloc[i]["label"]
                assert train_df.iloc[i]["label"] == result_df.iloc[i]["label"]
            # 将result_df中的预测结果值赋给train_df中的prediction
            train_df["prediction"] = result_df['outputs']
            # with open(f"../data/lap14/train_{args.train_idx}.json","w",encoding="utf-8") as train_json:
            #     json.dump(train_df,train_json,indent=4)
            train_df.to_json(f"../data/{args.dataset}/train_{args.train_idx}.json",orient='records')
if __name__ == "__main__":
    # 获得LLM的先验预测
    llm_pre_main()
