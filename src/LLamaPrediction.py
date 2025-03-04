import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ICL.ICL_templates import icl_instruction_prompt, label_space, get_input_template, feedback_prompt
from src.run_llm_hf_test import format_eval_output
from tool_utils.feedback_utils.data_utils import Tokenizer4LLM, ClassificationDataset, \
    ClassificationDataset_SemEvallap14
from tool_utils.feedback_utils.generate import LLMgenerate
from tool_utils.feedback_utils.util import seed_everything, readJson

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--icl_mode", type=str)
    parser.add_argument("--model", type=str, default="meta-llama2-7b-chat", help="LLM directory / name")
    parser.add_argument("--task", type=str, default="ATSC", choices=["ATSC", "SC", "EMO", "Irony", "Stance", "NLI"])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--train_examples", type=int, default=99, help="Number of training data")
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=750)
    parser.add_argument("--train_idx", type=int, default=1, help="Index of training data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--load_bit", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_ficl", type=bool, default=False)
    parser.add_argument("--run_icl", action="store_true", default=False)
    parser.add_argument("--run_pre", action="store_true", default=False)
    parser.add_argument("--wrong_exps", type=int, default=2)
    args = parser.parse_args()
    args.icl_mode = "MultiPerspectives"
    args.run_icl = True
    args.dataset = "SemEvallap14"
    seed_everything(seed=args.seed)
    # basic info
    print(f"LLM: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"ICL: {args.icl_mode}")
    print(f"Number of training data: {args.train_examples}")
    print(f"Number of Demos: {args.num_examples}")
    print("Loading Tokenizer")
    tokenizer = Tokenizer4LLM(args.max_len, args.model, llm=True)
    dataset_name = args.dataset
    task_name = args.task
    # load_dataset
    train_df = pd.read_json(f"../data/{dataset_name}/train_{args.train_idx}.json")
    # assert len(train_df) == args.train_examples
    test_df = pd.read_json(f"../data/{dataset_name}/test.json")

    # Select icl examples
    if args.icl_mode == "MultiPerspectives":
        example_ids = readJson(f"acos_MultiPerspectives/ATSC/SemEvallap14/average_ranks_per_input_sample.json")
    # Filter by current training data
    train_df_index = train_df.index.tolist()
    new_example_ids = dict()
    for k, v in example_ids.items():
        examples = [int(i) for i in v]
        new_example_ids[int(k)] = [i for i in examples if i in train_df_index]
    example_ids = new_example_ids
    if args.run_icl:
        prompt = icl_instruction_prompt[dataset_name]
        all_labels = label_space[dataset_name]
        TestDataset = ClassificationDataset(test_df, args, prompt, get_input_template, all_labels, tokenizer,
                                            args.icl_mode,
                                            examples=train_df, example_ids=example_ids, feedback_prompt=feedback_prompt)
        test_loader = DataLoader(TestDataset, batch_size=args.batch_size)
        print("Loading LLM ")
        LLM = LLMgenerate(args, tokenizer, all_labels=all_labels)
        # results
        rows = []
        print("Start Generation with the LLM")
        with torch.no_grad():
            for ii, d in enumerate(test_loader):
                print(f"Sample number：{ii}")
                if ii <= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)
                rows.extend(zip(d['index'], d['input_tokens'], d["labels"], output))
                print(output)
        output_file_path = f"../result/meta-llama2-7b-chat/{task_name}/{dataset_name}/train1_{args.icl_mode}_Multi_{args.num_examples}_icl.json"
        result_df = format_eval_output(rows, output_file_path)