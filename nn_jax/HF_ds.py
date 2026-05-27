from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import jax.numpy as jnp
import numpy as np
import os
from flax import nnx



def tokenize_function_single(sample,tokenizer):
    """This function tokenizes the sample with the tokenizer. It's used for the single sentence task

    Args:
        sample (_type_): dict
        tokenizer (_type_): hugging face tokenizer

    Returns:
        _type_: _description_
    """
    output = tokenizer(
        sample["sentence"], 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    seq_length = np.array(output["attention_mask"]).sum()
    # Return them with custom keys
    return {
        "input": output["input_ids"],
        "seq_len": seq_length,
        "label": sample["label"],
    }



def tokenize_function_sim_par(sample,tokenizer):
    """THis function does the same thing as the tokenize_function_single but for the similarity and paraphrase task"

    Args:
        sample (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Tokenize sentence 1 independently
    output1 = tokenizer(
        sample["sentence1"], 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    
    # Tokenize sentence 2 independently
    output2 = tokenizer(
        sample["sentence2"], 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    seq_length1 = np.array(output1["attention_mask"]).sum()
    seq_length2 = np.array(output2["attention_mask"]).sum()
    # Return them with custom keys
    return {
        "input1": output1["input_ids"],
        "input2": output2["input_ids"],
        "seq_len1" : seq_length1,
        "seq_len2" : seq_length2,
        "label": sample["label"]
    }




def rename_column_single(dataset,config):
    """Function that rename the columns to get a standard dataset for all config

    Args:
        dataset (_type_): HF datasets
        config (dict): config dict

    Returns:
        _type_: DatasetsDict
    """
    try:
        new_ds = dataset.rename_colum(config["input_name"][0],"sentence")
    except:
        new_ds = dataset
    return new_ds


def rename_column_multi(dataset,config):
    """This function does the same thing as rename_column_single but for the similiratiry and Paraphrase task

    Args:
        dataset (_type_): _description_
        config (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        new_ds = dataset.rename_column(config["input_name"][0],"sentence1")
        new_ds = new_ds.rename_column(config["input_name"][1],"sentence2")
    except:
        new_ds = dataset
    return new_ds



def get_datasets(config):
    """This function returns the datasets for the specified config

    Args:
        config (dict): Config training

    Returns:
        _type_: Datasets
    """
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    datasets= load_dataset("glue",config["name"],num_proc=16)
    if config["task"] == "Single Sentence":
        datasets_rename = rename_column_single(datasets,config)
        datasets_tokenized = datasets_rename.map(tokenize_function_single,remove_columns=["sentence"],fn_kwargs={"tokenizer":tokenizer})#,load_from_cache_file=False)
    elif config["task"] == "Similarity and Paraphrase":
        datasets_rename = rename_column_multi(datasets,config)
        datasets_tokenized = datasets_rename.map(tokenize_function_sim_par,remove_columns=["sentence1","sentence2"],fn_kwargs={"tokenizer":tokenizer})#,load_from_cache_file=False)
    return datasets_tokenized

def get_embedding_model(config):
    return nnx.Sequential(
        AutoModel.from_pretrained(config["tokenizer"]).embeddings.word_embeddings,)
        #AutoModel.from_pretrained(config["tokenizer"]).embeddings.LayerNorm,)