# adapted from https://github.com/thunlp/UltraChat/blob/main/train/ultrachat_dataset.py
import torch

def get_tokenizer_mapping_fn(
    tokenizer, cutoff_len, train_on_inputs=True, add_eos_token=True
):  
    IGNORE_INDEX = -100
    start_token = "\n"
    def tokenizer_mapping_fn(data_point):
        labels, tokenized_ids = [], []
        tags = [i for _ in range(len(data_point["data"])//2) for i in ["User", "Assistant"]]
        for i, c in enumerate(data_point["data"]):
            # model
            if i % 2 == 1:
                c_input = start_token + tags[i] + ": "
                tokenized = tokenizer(
                    c_input, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                )
                tokenized_ids += tokenized["input_ids"]
                if train_on_inputs:
                    labels += tokenized["input_ids"]
                else:
                    labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + tokenizer.eos_token
                tokenized = tokenizer(
                    c_generate, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                )
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]
            else:
                # user
                if i == 0:
                    c_new = tokenizer.bos_token + tags[i] + ": " + c + tokenizer.eos_token
                else:
                    c_new = start_token + tags[i] + ": " + c + tokenizer.eos_token
                tokenized = tokenizer(
                    c_new, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                )
                tokenized_ids += tokenized["input_ids"]
                if train_on_inputs:
                    labels += tokenized["input_ids"]
                else:
                    labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}
    return tokenizer_mapping_fn
