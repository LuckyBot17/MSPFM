import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ICL.ICL_templates import icl_instruction_prompt, label_space, get_input_template, feedback_prompt
from tool_utils.feedback_utils.data_utils import Tokenizer4LLM, ClassificationDataset
from tool_utils.feedback_utils.generate import LLMgenerate
from tool_utils.feedback_utils.util import seed_everything, readJson


def format_eval_output(rows,output_file_path):
    # 解压 rows
    idx, tokens, labels, outputs = zip(*rows)
    # 将每个字段转换为一维列表
    idx = np.vstack(idx).reshape(-1).tolist()
    tokens = np.vstack(tokens).reshape(-1).tolist()
    labels = np.vstack(labels).reshape(-1).tolist()
    outputs = np.vstack(outputs).reshape(-1).tolist()

    # 组合成最终的 JSON 数据
    results = [{"id": i, "input_all_tokens": t, "label": l, "outputs": o}
               for i, t, l, o in zip(idx, tokens, labels, outputs)]
    if output_file_path:
        with open(output_file_path,"w") as f:
            json.dump(results,f,indent=4)

def llm_test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42,help="Random seed")
    parser.add_argument("--icl_mode",type=str)
    parser.add_argument( "--model",type=str,default="meta-llama2-13b-chat",help="LLM directory / name")
    parser.add_argument("--task",type=str,default="ATSC",choices=["ATSC", "SC", "EMO", "Irony", "Stance", "NLI"])
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--train_examples",type=int,default=99,help="Number of training data")
    parser.add_argument("--num_examples",type=int,default=4)
    parser.add_argument("--max_len",type=int,default=750)
    parser.add_argument("--train_idx", type=int, default=1, help="Index of training data")
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--load_bit", type=str, default="fp16")
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--run_ficl",type=bool,default=False)
    parser.add_argument("--run_icl", action="store_true", default=False)
    parser.add_argument("--run_pre", action="store_true", default=False)
    parser.add_argument("--wrong_exps", type=int, default=2)
    args = parser.parse_args()
    args.icl_mode = "MultiPerspectives"
    args.run_ficl = True
    args.dataset = "Clothing"
    seed_everything(seed=args.seed)
    # basic info
    print(f"LLM: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"ICL: {args.icl_mode}")
    print(f"Number of training data: {args.train_examples}")
    print(f"Number of Demos: {args.num_examples}")
    print("Loading Tokenizer")
    tokenizer = Tokenizer4LLM(args.max_len,args.model,llm=True)
    dataset_name = args.dataset
    task_name = args.task

    #load_dataset
    train_df = pd.read_json(f"../data/{dataset_name}/train_{args.train_idx}.json")
    #train_df = train_df[:args.train_examples]
    #assert len(train_df) == args.train_examples
    test_df = pd.read_json(f"../data/{dataset_name}/test.json")

    #Select icl examples
    if args.icl_mode == "MultiPerspectives":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/average_ranks_per_input_sample.json")
    # Filter by current training data
    train_df_index = train_df.index.tolist()
    new_example_ids = dict()
    for k,v in example_ids.items():
        examples = [int(i) for i in v]
        new_example_ids[int(k)] = [i for i in examples if i in train_df_index]
    example_ids = new_example_ids
    if args.run_ficl:
        prompt = icl_instruction_prompt[dataset_name]#"Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
        all_labels = label_space[dataset_name]#["NEG", "NEU", "POS"]
        TestDataset = ClassificationDataset(test_df,args,prompt,get_input_template,all_labels,tokenizer,args.icl_mode,
                                            examples=train_df,example_ids=example_ids,feedback_prompt=feedback_prompt)
        test_loader = DataLoader(TestDataset,batch_size=args.batch_size)
        print("Loading LLM ")
        LLM = LLMgenerate(args,tokenizer,all_labels=all_labels)
        #results
        rows = []
        print("Start Generation with the LLM")
        with torch.no_grad():
            for ii,d in enumerate(test_loader):
                print(f"Sample number：{ii}")
                if ii<= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)
                rows.extend(zip(d['index'],d['input_tokens'],d["labels"],output))
                print(output)
        output_file_path = f"../result/{args.model}/{task_name}/{dataset_name}/train{args.train_idx}_{args.icl_mode}_{args.num_examples}_ficl.json"
        result_df = format_eval_output(rows,output_file_path)

if __name__ == '__main__':
    llm_test_main()