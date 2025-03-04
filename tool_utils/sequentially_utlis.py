import numpy as np
import torch
from torch.utils.data import TensorDataset


class InputFeatures(object):
    def __init__(self, input_ids=None, input_masks=None, input_seg_ids=None, label_id=None, ):
        self.input_ids = input_ids
        self.input_masks = input_masks

def convert_examples_to_features(examples,max_seq_length,tokenizer,mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token

    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    def inputIdMaskSegment(tmp_text=None):
        tokens = []
        # for tok in tmp_text.split():
        #     tokens.extend(tokenizer.wordpiece_tokenizer.tokenize(tok))
        tokens.extend(tokenizer.tokenize(tmp_text))
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = [cls_token] + tokens + [sep_token]
        # 将 tokens 转换为模型可理解的ID列表
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        # input_ids += ([pad_token_id] * padding_length)
        # input_mask += ([0] * padding_length)
        input_ids = np.pad(input_ids,(0,padding_length),constant_values=pad_token_id).tolist()
        input_mask = np.pad(input_mask,(0,padding_length),constant_values=0).tolist()
        assert len(input_ids) == len(input_mask) == max_seq_length
        return input_ids, input_mask

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids, input_masks = inputIdMaskSegment(tmp_text=example.text.lower())
        features.append(InputFeatures(input_ids=input_ids, input_masks=input_masks, ))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_masks)
    return dataset