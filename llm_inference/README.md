## Offline inference with AxoNN

Offline inference refers to the process of running machine learning models locally using a predefined 
set of prompts or inputs. This means that the entire inference process is self-contained and operates 
independently on the local machine. Unlike online inference, it is assumed that you have your all of your 
inference prompts ready before inference begins. 


This example runs offline inference on large language models (LLMs) available on 
Huggingface Transformers. It utilizes prompts sampled from the Alpaca Eval dataset to measure performance. 
The key feature of this example is demonstrating multi-GPU inference with tensor parallelism using AxoNN.

### Features 

- Multi-GPU inference with Tensor Parallelism: Shard large models like `meta-llama/Llama-2-70b-chat-hf` on multiple GPUs easily
- Compatibility with Huggingface Transformers: Works with a large set of models on huggingface transformers 

## Install dependencies

This example depends on `transformers` for downloading and initializing LLMs and `datasets` for downloading the 
alpaca eval dataset. To install these, run the following command - 

```
pip install transformers datasets
```

For tensor parallelism, we use [AxoNN](https://github.com/axonn-ai/axonn). To install AxoNN, run the following commands - 

```
git clone https://github.com/axonn-ai/axonn
cd axonn
pip install -e .
```

That's it! Now you're ready to run this example.

## Running the example

### Single node, multiple GPUs

Say you want to run this example on `meta-llama/Llama-2-7b-chat-hf` with 10 prompts 
sampled from the AlpacaEval dataset. Let us also assume that you want to use tensor parallelism 
across four GPUs on a single node. Then here's what you will run - 


```
torchrun --nproc_per_node 4 infer.py --model_id "meta-llama/Llama-2-7b-chat-hf" --num-prompts 10 --seed 123456 --static-kv-cache --dtype "fp16"
```

Now, let's describe each of the arguments:

- `--model_id`: Specifies the name of the Huggingface Transformers model you want to run. This argument allows you to select which pre-trained model to use for offline inference.

- `--num-prompts`: Specifies the number of prompts to use from the Alpaca Eval dataset. This argument controls the size of the input data used for evaluation.

- `--seed`: Specifies the random seed to use for reproducibility. This argument ensures that the randomization in the example is consistent across runs.

- `--static-kv-cache`: This is a flag argument. When enabled, it instructs the code to use a static key-value cache for faster inference.

### Multi node, multiple GPUs




