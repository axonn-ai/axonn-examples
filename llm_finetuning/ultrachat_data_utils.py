def get_tokenizer_mapping_fn(
    tokenizer, cutoff_len, train_on_inputs=True, add_eos_token=True
):  
    IGNORE_INDEX = -100
    start_token = "\n"
    def tokenizer_mapping_fn(data_point):
        labels, tokenized_ids = [], []
        tags = [i for _ in range(len(data_point["data"])//2) for i in ["User", "Assistant"]]
        for i, data in enumerate(data_point["data"]):
            # model
            if i % 2 == 1:
                model_prompt = start_token + tags[i] + ": "
                tokenized = tokenizer(
                    model_prompt, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                    return_tensors=None,
                )
                tokenized_ids += tokenized["input_ids"]
                if train_on_inputs:
                    labels += tokenized["input_ids"]
                else:
                    labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                model_response = data + tokenizer.eos_token
                tokenized = tokenizer(
                    model_response, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                    return_tensors=None,
                )
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]
            else:
                # user
                if i == 0:
                    user_prompt = tags[i] + ": " + data + tokenizer.eos_token
                else:
                    user_prompt = start_token + tags[i] + ": " + data + tokenizer.eos_token
                tokenized = tokenizer(
                    user_prompt, 
                    truncation=True, 
                    max_length=cutoff_len, 
                    padding=False,
                    return_tensors=None,
                )
                tokenized_ids += tokenized["input_ids"]
                if train_on_inputs:
                    labels += tokenized["input_ids"]
                else:
                    labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": tokenized_ids, "labels": labels}
    return tokenizer_mapping_fn
