from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from axonn.models.transformers import parallelize 
from axonn import axonn as ax
import torch
import random
import numpy as np
from argparse import ArgumentParser
from contextlib import nullcontext

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

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", type=str,
                       help="name of huggingface transformers model you want to run")
    parser.add_argument("--seed", type=int, default=123456, 
                        help="random seed")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                        help="data type for running inference", default="fp16")
    parser.add_argument("--use-flash-attention", action='store_true',
                        help="Use Flash Attention for faster training")
    parser.add_argument("--global-batch-size", type=int, default=4, 
                        help="Global Batch Size")
    parser.add_argument("--gradient-acc-steps", type=int, default=1, 
                        help="Gradient Accumulation Steps")
    parser.add_argument("--sequence-length", type=int, default=512, 
                        help="Sequence Length")
    parser.add_argument("--num-iters", type=int, default=1000, 
                        help="Number of Training Iterations")
    parser.add_argument("--disable-axonn", action='store_false', dest='use_axonn',
                        help="Disable AxoNN's Tensor Paralellism")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Interval for logging train loss")
    return parser

dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}


def get_tokenized_wikitext(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    return encodings

def load_prompts(encodings, batch_size, prompt_length):
    total_tokens = encodings.input_ids.shape[1]
    input_ids = []
    for _ in range(batch_size):
        start_index = min(random.randint(0, total_tokens), total_tokens - prompt_length)
        tokens = encodings.input_ids[:, start_index : start_index + prompt_length].reshape(1, prompt_length)
        input_ids.append(tokens)
    input_ids = torch.cat(input_ids, dim=0)
    return input_ids

def pretty_log(iteration,
               total_train_iters,
               train_loss,
               elapsed_time_per_iteration,
               learning_rate,
):

    log_string = '> global batch {:8d}/{:8d} |'.format(
        iteration, total_train_iters)
    log_string += ' elapsed time per global batch (ms): {:.1f} |'.format(
        elapsed_time_per_iteration)
    log_string += ' learning rate: {:.3E} |'.format(learning_rate)
    log_string += ' loss: {:.5f} |'.format(train_loss)
    curr_mem =  torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_mem =  torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    log_string += ' memory used by tensors {:.3f} GB (peak {:.3f} GB) '.format(curr_mem, peak_mem)
    return log_string

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    init_everything()
    set_seed(args.seed)
    dtype = dtype_map[args.dtype]

    with parallelize(args.model_id, enabled=args.use_axonn):
        model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                         torch_dtype=dtype, 
                                                         attn_implementation='eager' if not args.use_flash_attention else "flash_attention_2").to('cuda').float()

    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenized_dataset = get_tokenized_wikitext(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    loss_fn = torch.nn.CrossEntropyLoss()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for iter_no in range(args.num_iters):
        start_event.record()
        batch_loss = 0
        for grad_acc_step in range(args.gradient_acc_steps):
            batch = load_prompts(tokenized_dataset, args.global_batch_size // args.gradient_acc_steps, args.sequence_length+1).cuda()
            x, y = batch[:,:-1], batch[:,1:]
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                output = model(x)
                logits = output["logits"]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), 
                               y.reshape(-1))
            scaler.scale(loss / args.gradient_acc_steps).backward()
            batch_loss += loss / args.gradient_acc_steps
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        end_event.record()
        if torch.distributed.get_rank() == 0 and (iter_no % args.log_interval==0):
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) 
            log_string = pretty_log(iter_no, args.num_iters, batch_loss.item(), elapsed_time, learning_rate=optimizer.param_groups[0]['lr'])
            print(log_string)

 



