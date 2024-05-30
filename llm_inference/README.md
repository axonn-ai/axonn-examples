## Offline inference with AxoNN

Offline inference refers to the process of running machine learning models locally using a predefined 
set of prompts or inputs. This means that the entire inference process is self-contained and operates 
independently on the local machine. Unlike online inference, which relies on continuous communication with 
remote servers (e.g., ChatGPT), offline inference runs entirely on local hardware without external dependencies.


This benchmark tool evaluates the performance of large language models (LLMs) using offline inference 
with Huggingface Transformers. It utilizes prompts sampled from the Alpaca Eval dataset to measure performance. 
The key feature of this benchmark is demonstrating multi-GPU inference with tensor parallelism using AxoNN.

### Features 

- Multi-GPU inference with Tensor Parallelism: Shard large models like `meta-llama/Llama-2-70b-chat-hf` on multiple GPUs

## How to run?

To run on a single node with four GPUs
`torchrun --nproc_per_node 4 infer.py`
