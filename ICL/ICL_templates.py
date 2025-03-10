import pandas as pd
label_space = {
    # SC
    "tweet": ["negative", "neutral", "positive"],  # 0,1,2
    "fin": ["negative", "neutral", "positive"],  # 0,1,2
    "poem": ['negative', 'positive', 'neutral'],
    "sst2": ['negative', 'positive'],
    # ATSC
    "SemEvalrest14": ["negative", "neutral", "positive"],  # 0,1,2
    "SemEvallap14":["negative", "neutral", "positive"],#0,1,2
    "lap14": ["NEG", "NEU", "POS"],  # 0,1,2
    "Books":["NEG", "NEU", "POS"],
    "Clothing":["NEG","NEU","POS"],
    "twitter": ["negative", "neutral", "positive"],  # 0,1,2
    # EMO
    "tweet_emo": ['anger', 'joy', 'optimism', 'sadness'],
    "emocontext": ['others', 'happy', 'sad', 'angry'],
    # Irony
    "tweet_irony": ['non-irony', 'irony'],
    # Topic
    "agnews": ['World', 'Sports', 'Business', 'Technology'],
    # Stance
    "pstance": ['against', 'favor'],
    # NLI
    "mnli": ['entailment', 'neutral', 'contradiction']
}

feedback_prompt = {"wrong": "You are wrong ! Make sure your prediction is accurate.\n\n",
                   "correct": "You are correct ! Stay determined and keep moving forward.\n\n"}

icl_instruction_prompt = {
    # SC
    "poem":
        "Recognize the sentiment of the sentence. Here are some examples:\n\n"
    ,
    "tweet":
        "Recognize the sentiment of the sentence. Here are some examples:\n\n"
    ,
    "fin":
        "Recognize the sentiment of the sentence. Here are some examples:\n\n"
    ,
    "sst2":
        "Recognize the sentiment of the sentence. Here are some examples:\n\n"
    ,
    # ATSC
    "SemEvalrest14":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    "lap14":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    "Books":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    "Clothing":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    "twitter":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    "SemEvallap14":
        "Recognize the sentiment polarity for the given aspect term in the sentence. Here are some examples:\n\n"
    ,
    # EMO
    "tweet_emo":
        "Recognize the emotion of the sentence. Here are some examples:\n\n"
    ,
    "emocontext":
        "Recognize the emotion of the sentence. Here are some examples:\n\n"
    ,
    # Irony
    "tweet_irony":
        "Determine whether the sentence is ironic or not. Here are some examples:\n\n"
    ,
    # Stance
    "pstance":
        "Recognize the stance of the sentence to the given target. Here are some examples:\n\n"
    ,
    # NLI
    "mnli":
        "Recognize textual entailment between the 2 texts. Here are some examples:\n\n"
    ,

}


def get_input_template(example, dataset_name):
    if dataset_name in ["tweet", "sst2", "poem", "fin", "tweet_emo", "emocontext", "tweet_irony"]:
        return f"Sentence: {example['sentence']}\n"
    if dataset_name in ["SemEvalrest14","twitter","SemEvallap14"]:
        return f"Sentence: {example['sentence']} What is the sentiment polarity of the aspect {example['aspect']} ?\n"
    if dataset_name in ['lap14',"Books",'Clothing']:
        quadruple = example['quadruples'][0]
        return f"Sentence: {example['raw_words']} What is the sentiment polarity of the aspect {quadruple['aspect']['term']} ?\n"
    if dataset_name in ["pstance"]:
        return f"Sentence: {example['sentence']} What is the attitude of sentence toward target {example['target']} ?\n"
    if dataset_name in ["mnli"]:
        return f"Premise: {example['text1']}\nHypothesis: {example['text2']}\n"

