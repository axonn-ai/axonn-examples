## Offline inference with AxoNN

Offline inference refers to the process of running machine learning models locally using a predefined 
set of prompts or inputs. This means that the entire inference process is self-contained and operates 
independently on the local machine. Unlike online inference, which relies on continuous communication with 
remote servers (e.g., ChatGPT), offline inference runs entirely on local hardware without external dependencies.





To run on a single node with four GPUs
`torchrun --nproc_per_node 4 infer.py`
