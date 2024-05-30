from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache 
from datasets import load_dataset
from axonn.models.transformers import parallelize 
from axonn import axonn as ax
import torch
import random
import numpy as np
from argparse import ArgumentParser

OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

def init_everything():
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Going to distribute the model over {world_size} GPUs")
    ax.init(G_data=1, G_inter=1, G_intra_r=world_size, G_intra_c=1, G_intra_d=1)

def set_seed(seed=123456):
    # Extremely important for AxoNN
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def adjust_attention_heads_for_axonn(config):
    #config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
    world_size = torch.distributed.get_world_size()
    if config.num_key_value_heads is None:
        config.num_attention_heads //= world_size
    else:
        config.num_key_value_heads //= world_size

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-2-7b-chat-hf", type=str,
                       help="name of huggingface transformers model you want to run")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="number of prompts to use from alpaca eval dataset")
    parser.add_argument("--seed", type=int, default=123456, 
                        help="random seed")
    parser.add_argument("--static-kv-cache", action='store_true',
                        help="use static kv-cache for faster inference")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                        help="data type for running inference", default="fp16")
    return parser

dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}

if  __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # since we will not run backward passes
    # we disable any caching of activations
    torch.set_grad_enabled(False)
    init_everything()
    set_seed(args.seed)
    dtype = dtype_map[args.dtype] 

    with parallelize(args.model_id):
        model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                     torch_dtype=dtype, 
                                                     attn_implementation='eager').to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)
    adjust_attention_heads_for_axonn(config)

    dataset = load_dataset("tatsu-lab/alpaca_eval")
    prompts = dataset["eval"]["instruction"]
    random.shuffle(prompts)
    prompts = prompts[:args.num_prompts]

    total_generated_tokens = 0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    generations = []

    max_new_tokens = 1024
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if args.static_kv_cache:
            kv_cache = StaticCache(config=config,
                                   max_batch_size=1, 
                                   max_cache_len=input_ids.numel()+max_new_tokens,
                                   dtype=dtype,
                                   device='cuda')
        else:
            kv_cache = None
        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model.generate(input_ids.cuda(), 
                                     do_sample=True, 
                                     max_new_tokens=max_new_tokens,
                                     past_key_values=kv_cache,
                                     use_cache=False)

        generated_tokens = outputs.numel() -  input_ids.numel()
        total_generated_tokens += generated_tokens

        generations.append(outputs[:, input_ids.numel():])

    end_event.record()
    
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)
    tput = total_generated_tokens * 1000 / total_time

    if torch.distributed.get_rank() == 0:
        for prompt, generation in zip(prompts,generations):
            print(f"{OKBLUE}[PROMPT]: {prompt}{ENDC}")
            print(f"{OKGREEN}[GENERATION]: = {tokenizer.batch_decode(generation)[0]}{ENDC}")
            print("=====")
        print(f"Tput = {tput} generated tokens / second")

