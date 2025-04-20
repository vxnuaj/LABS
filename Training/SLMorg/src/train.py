import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os

from dataloader import get_dataloader, get_data
from utils import get_config, get_custom_scheduler, load_checkpoint, train, get_model_parallel, setup_parallel, init_wandb, _save_config
from slm import SLM

model_config = get_config(type_='model')
opt_config = get_config(type_='opt')
lr_config = get_config(type_='lr')
loss_config = get_config(type_='loss')
train_config = get_config(type_='train')
dataloader_config = get_config(type_='dataloader')
val_dataloader_func_config = get_config(type_='val_dataloader')
parallel_config = get_config(type_='parallel')
wandb_config = get_config(type_='wandb')
generate_val_config = get_config(type_ = 'generate_val')

_save_config(
    input_dir = 'src/config/',
    output_dir = train_config['save_checkpoint_path'],
    run = train_config['run']
    )

track_grad_norm = train_config['track_grad_norm']
clip_grad_norm = train_config['clip_grad_norm']
max_grad_norm = train_config['max_grad_norm']
total_epochs = train_config['total_epochs']
val_steps = train_config['val_steps']
mixed_precision = train_config['mixed_precision']
save_checkpoint_steps = train_config['save_checkpoint_steps']
save_checkpoint_path = train_config['save_checkpoint_path']
load_checkpoint_path = train_config['load_checkpoint_path']
batch_size = train_config['batch_size']
sample_model_val = train_config['sample_model_val']

parallel = train_config['parallel']
seed = train_config['seed']
pad_token_id = train_config['pad_token_id']

assert pad_token_id == 1

train_tensor_path = dataloader_config['train_tensor_path']
list_ = dataloader_config['list_']
num_workers = dataloader_config['num_workers']
shuffle = dataloader_config['shuffle']
pin_memory = dataloader_config['pin_memory']

sample_model_str = generate_val_config['sample_model_str']
tokenizer_path = generate_val_config['tokenizer_path']

wandb_ = wandb_config['wandb_']

_model_key = 'model_state_dict'
_optim_key = 'optim_state_dict'
_scheduler_key = 'scheduler_state_dict'

if parallel:
    parallelism_type = parallel_config['parallelism_type']
    backend = parallel_config['backend']
    min_num_params = parallel_config['min_num_params']
    type_ = parallel_config.get('type_') if parallelism_type == 'fsdp' else None
else:
    parallelism_type = None
    backend = None
    min_num_params = None
    type_ = None

parallel_dict = setup_parallel(parallelism_type=parallelism_type, backend=backend, sleep=3)
device = parallel_dict['device']
world_size = parallel_dict['world_size']
rank = parallel_dict['rank']
local_rank = parallel_dict.get('local_rank')

torch.manual_seed(seed + (rank if parallelism_type in ['fsdp', 'ddp'] else 0))

model = SLM(**model_config)
model = get_model_parallel(model, parallelism_type=parallelism_type, device=device, type_=type_,
                          min_num_params=min_num_params, backend=backend)
optim = optim.AdamW(model.parameters(), **opt_config)
scheduler = get_custom_scheduler(optim, **lr_config)
criterion = nn.CrossEntropyLoss(**loss_config)

if load_checkpoint_path:
    model, optim, checkpoint_epoch, global_steps, _, checkpoint_batch_idx = load_checkpoint(
        model, optim, scheduler, load_checkpoint_path, _model_key, _optim_key, _scheduler_key, parallelism_type)
else:
    checkpoint_epoch = 0
    checkpoint_batch_idx = 0
    global_steps = 0

X, y = get_data(path=train_tensor_path)

dataloader = get_dataloader(X=X, y=y, batch_size=batch_size, num_workers=num_workers,
                            shuffle=shuffle, pin_memory=pin_memory, parallelism_type=parallelism_type, rank=rank)

if rank == 0:
    os.makedirs(save_checkpoint_path, exist_ok=True)

if wandb_ and rank == 0:
    init_wandb(**wandb_config)

train(
    model=model,
    optim=optim,
    scheduler=scheduler,
    criterion=criterion,
    dataloader=dataloader,
    device=device,
    total_epochs=total_epochs,
    val_steps = val_steps,
    mixed_precision=mixed_precision,
    rank=rank,
    world_size=world_size,
    parallelism_type=parallelism_type,
    checkpoint_epoch=checkpoint_epoch,
    checkpoint_batch_idx=checkpoint_batch_idx,
    clip_grad_norm = clip_grad_norm,
    max_grad_norm = max_grad_norm,
    track_grad_norm = track_grad_norm,
    global_steps=global_steps,
    val_dataloader_func_config=val_dataloader_func_config,
    generate_val_config = generate_val_config,
    save_checkpoint_steps=save_checkpoint_steps,
    save_checkpoint_path=save_checkpoint_path,
    tokenizer_path = tokenizer_path,
    sample_model_val = sample_model_val,
    sample_model_str = sample_model_str,
    pad_token_id = pad_token_id,
    wandb_=wandb_ and rank == 0 
)

if parallelism_type in ['ddp', 'fsdp']:
    dist.destroy_process_group()