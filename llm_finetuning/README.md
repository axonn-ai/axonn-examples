

```
torchrun --nproc_per_node 4 finetune.py --dtype bf16 \ 
                                --use-flash-attention \
                                --model_id openlm-research/open_llama_3b_v2 \ 
                                --global-batch-size 32 \
                                --gradient-acc-steps 4 \
                                --log-interval 1
```

