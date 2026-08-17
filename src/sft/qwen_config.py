import os
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# 1. Settings
# =========================================================
pretrained_model_name_or_path = 'Qwen/Qwen2.5-7B'
data_path = os.getenv('XTUNER_DATA_PATH', 'data/training_sets/selected_kmeans_max_n20000.jsonl')

max_length = 32768  
pack_to_max_length = True
batch_size = 4
accumulative_counts = 8
lr = 3e-5
min_lr = 1e-6
max_epochs = 4

work_dir = os.getenv('XTUNER_WORK_DIR', './output/models/qwen_sft')

# =========================================================
# 2. Dataset Mapping Function 
# =========================================================
def process_custom_dataset(example):
    """
    Input (Data): {'dialogs': [{'role': 'user', ...}, ...], 'id_ddm': ...}
    Output (XTuner):   {'conversation': [{'role': 'user', ...}, ...]}
    """
    return {'conversation': example['dialogs']}

# =========================================================
# 3. Model & Tokenizer
# =========================================================
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16)

# =========================================================
# 4. Dataset Config
# =========================================================
dataset = dict(
    type='xtuner.dataset.process_hf_dataset',
    dataset=dict(
        type='json',
        data_files=dict(train=data_path),
        split='train'),
    tokenizer=tokenizer,
    max_length=max_length,
    
    dataset_map_fn=process_custom_dataset, 
    
    template_map_fn=dict(
        type='xtuner.dataset.map_fns.template_map_fn_factory',
        template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat'),
    remove_unused_columns=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'))

# =========================================================
# 5. Optimization & Scheduler
# =========================================================
optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype='bfloat16',
    optimizer=dict(type=AdamW, lr=lr, weight_decay=0.05))

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=min_lr / lr,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=min_lr,
        by_epoch=True,
        begin=1,
        end=max_epochs,
        convert_to_iter_based=True)
]

# =========================================================
# 6. Runtime Hooks
# =========================================================
train_cfg = dict(type='TrainLoop', max_epochs=max_epochs, val_interval=1)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=10),
    
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=1),
    param_scheduler=dict(type=ParamSchedulerHook),
    dist_sampler_seed=dict(type=DistSamplerSeedHook),
)

visualizer = None
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
resume = False
load_from = None