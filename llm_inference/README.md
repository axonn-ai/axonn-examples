## Offline inference with AxoNN

Offline inference refers to the process of running machine learning models locally using a predefined 
set of prompts or inputs. This means that the entire inference process is self-contained and operates 
independently on the local machine. Unlike online inference, which relies on continuous communication with 
remote servers (e.g., ChatGPT), offline inference runs entirely on local hardware without external dependencies.


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



To run on a single node with four GPUs
`torchrun --nproc_per_node 4 infer.py`
