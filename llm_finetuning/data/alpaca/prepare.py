import os 
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from alpaca_data_utils import get_tokenizer_mapping_fn

dataset_id = "tatsu-lab/alpaca"
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
max_seq_len = 256


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"  

def get_tokenized_dataset(tokenizer, sequence_length):
    dataset = dataset_id.split('/')[1]
    assert dataset in ["alpaca", "ultrachat"]
        
    data = load_dataset(dataset_id)
    mapping_fn = get_tokenizer_mapping_fn(
        tokenizer, cutoff_len=sequence_length, train_on_inputs=False
    )
    train_data = (
        data["train"]
        .shuffle()
        .map(mapping_fn, remove_columns=data["train"].column_names, num_proc=os.cpu_count() // 2 if os.cpu_count() // 2 else 8)
    )
    return train_data

if __name__ == "__main__":
    train_data = get_tokenized_dataset(tokenizer, max_seq_len)
    train_data.save_to_disk("./")
