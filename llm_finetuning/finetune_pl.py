from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForSeq2Seq
from datasets import load_dataset
from axonn.models.transformers import parallelize 
from axonn import axonn as ax
import torch
import random
import numpy as np
from argparse import ArgumentParser
from contextlib import nullcontext
from data_utils import get_tokenizer_mapping_fn
from torch.utils.data import DataLoader
from axonn.checkpoint import save

from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from axonn.intra_layer import optimize_communication, clear_weights_cache

def init_everything(dtype, num_nodes=4):
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    pl_strategy = AxonnStrategy(
                G_intra_d = world_size
    )
    fabric = Fabric(strategy=pl_strategy, devices=4, num_nodes=num_nodes, precision=dtype)
    fabric.launch()

    if torch.distributed.get_rank() == 0:
        print(f"Going to distribute the model over {world_size} GPUs")

    return fabric

def set_seed(seed=123456):
    seed_everything(seed)

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", type=str,
                       help="name of huggingface transformers model you want to run")
    parser.add_argument("--seed", type=int, default=123456, 
                        help="random seed")
    parser.add_argument("--dtype", choices=["bf16-mixed", "16-mixed", "32"],
                        help="data type for running inference", default="bf16-mixed")
    parser.add_argument("--use-flash-attention", action='store_true',
                        help="Use Flash Attention for faster training")
    parser.add_argument("--global-batch-size", type=int, default=4, 
                        help="Global Batch Size")
    parser.add_argument("--gradient-acc-steps", type=int, default=1, 
                        help="Gradient Accumulation Steps")
    parser.add_argument("--sequence-length", type=int, default=256, 
                        help="Sequence Length")
    parser.add_argument("--disable-axonn", action='store_false', dest='use_axonn',
                        help="Disable AxoNN's Tensor Paralellism")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Interval for logging train loss")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save model weights after every --save-every iterations")
    parser.add_argument("--wandb-log", action='store_true',
                        help="Use Wandb for logging")
    parser.add_argument("--wandb-project", type=str,
                        help="Name of wandb project (only necessary when --wandb-log is used)")
    parser.add_argument("--wandb-run-name", type=str,
                        help="Name of wandb run (only necessary when --wandb-log is used)")
    return parser



def get_tokenized_dataset(tokenizer, sequence_length=256):
    data = load_dataset("tatsu-lab/alpaca")
    mapping_fn = get_tokenizer_mapping_fn(tokenizer, cutoff_len=sequence_length, train_on_inputs=False)
    train_data = data["train"].shuffle().map(mapping_fn, remove_columns=data["train"].column_names)
    return train_data

def pretty_log(iteration,
               total_train_iters,
               train_loss,
               elapsed_time_per_iteration,
               learning_rate,
               wandb_log=False
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
    if wandb_log:
        import wandb
        wandb.log(
            {
                "iter": iteration,
                "train/loss": train_loss,
                "lr": learning_rate,
                "peak_memory": peak_mem,
                "state_memory": curr_mem,
                "iteration_time": elapsed_time_per_iteration
            }
        )
    return log_string


dtype_map = {
        "bf16-mixed": torch.bfloat16,
        "16-mixed": torch.float16,
        "32": torch.float32
}

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    fabric = init_everything(args.dtype)
    set_seed(args.seed)
    if args.wandb_log and torch.distributed.get_rank() == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    if args.use_axonn:
        with parallelize(args.model_id):
            model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                         attn_implementation='eager' if not args.use_flash_attention else "flash_attention_2").to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                         attn_implementation='eager' if not args.use_flash_attention else "flash_attention_2").to('cuda')

    model.train()
    model.gradient_checkpointing_enable()


    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=1e-5, 
                                  betas=(0.9, 0.95), 
                                  eps=1e-5,
                                  weight_decay=0.0)

    
    model, optimizer = fabric.setup(model, optimizer)
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token_id = (
        0  
    )
    tokenizer.padding_side = "left"  
    tokenized_dataset = get_tokenized_dataset(tokenizer, args.sequence_length)
    dataloader = DataLoader(
                                tokenized_dataset, 
                                batch_size=args.global_batch_size // args.gradient_acc_steps // torch.distributed.get_world_size(), 
                                collate_fn=DataCollatorForSeq2Seq(
                                        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
                            ) 

    dataloader = fabric.setup_dataloaders(dataloader)

    iters_per_epoch = len(dataloader) // args.gradient_acc_steps
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters_per_epoch * args.num_epochs)
    warmup_iters = 100
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_iters
            )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )

    loss_fn = torch.nn.CrossEntropyLoss()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iter_no = 0
    for epoch_no in range(args.num_epochs):
        microbatch_no = 0
        start_event.record()
        batch_loss = 0
        for batch in dataloader:
            input_ids, labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
            input_ids, labels, attention_mask = input_ids.cuda(), labels.cuda(), attention_mask.cuda()
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            labels = labels[:, 1:]
            with optimize_communication(True, True, True, model):
                output = model(input_ids = input_ids, attention_mask=attention_mask)
                logits = output["logits"]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), 
                               labels.reshape(-1))
                fabric.backward(loss / args.gradient_acc_steps, model=model)
            clear_weights_cache()
            batch_loss += (loss / args.gradient_acc_steps).item()
            microbatch_no += 1

            if microbatch_no == args.gradient_acc_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                iter_no += 1
                end_event.record()
                if torch.distributed.get_rank() == 0 and (iter_no % args.log_interval==0):
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) 
                    log_string = pretty_log(iter_no, len(dataloader)*args.num_epochs // args.gradient_acc_steps, 
                                            batch_loss, elapsed_time, learning_rate=optimizer.param_groups[0]['lr'],
                                            wandb_log=args.wandb_log)
                    print(log_string)

                microbatch_no = 0
                batch_loss = 0
                start_event.record()
                state = {
                        "iter_no": iter_no,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict()
                        }
                
    save(state, checkpoint_folder="./ckpt", checkpoint_name=f"iter_{iter_no}")

 



