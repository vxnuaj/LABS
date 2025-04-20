import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import torch.nn.functional as F

import sys
import random
import wandb
import json
import time
import warnings
import functools
import os
import gc

from typing import List, Union
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerFast
from blocks import TransformerBlock
from dataloader import get_dataloader
from tqdm import tqdm

def train(
    model,
    optim,
    scheduler,
    criterion,
    dataloader,
    device,
    val_steps:int,
    autocast_dtype: torch.dtype = None,
    mixed_precision=False,
    total_epochs=None,
    clip_grad_norm=False,
    max_grad_norm: int = 5,
    track_grad_norm:bool = False,
    rank=None,
    parallelism_type: str = None,
    checkpoint_epoch: int = 0,
    checkpoint_batch_idx: int = 0,
    global_steps: int = 0,
    sample_model_val:bool = False,
    sample_model_str:Union[str, List[str]] = None,
    tokenizer_path:str = None,
    save_checkpoint_steps: int = None,
    save_checkpoint_path: str = None,
    _model_key='model_state_dict',
    _optim_key='optim_state_dict',
    _scheduler_key='scheduler_state_dict',
    wandb_=False,
    pad_token_id=None,
    val_dataloader_func_config:dict = None,
    generate_val_config:dict = None,
    **kwargs
    ):
    
    """
    Trains a PyTorch model over multiple epochs with support for mixed precision, distributed training,
    gradient clipping, validation, checkpointing, and logging.

    Args:
        model (torch.nn.Module): The model to train.
        optim (torch.optim.Optimizer): Optimizer used for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (callable): Loss function that computes per-element losses.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
        device (torch.device): Device on which to perform computations.
        val_steps (int): Number of training steps between validations.
        autocast_dtype (torch.dtype, optional): Data type for automatic mixed precision.
        mixed_precision (bool, optional): Whether to use mixed precision training.
        total_epochs (int, optional): Total number of epochs to run.
        clip_grad_norm (bool, optional): Whether to enable gradient clipping.
        max_grad_norm (int, optional): Maximum norm for gradient clipping.
        track_grad_norm (bool, optional): Whether to record gradient norms.
        rank (int, optional): Rank of the current process in distributed setup.
        parallelism_type (str, optional): Parallelism method: 'fsdp', 'ddp', or 'dp'.
        checkpoint_epoch (int, optional): Epoch index at which to resume training.
        checkpoint_batch_idx (int, optional): Batch index within epoch to resume from.
        global_steps (int, optional): Initial global step count.
        sample_model_val (bool, optional): Whether to generate samples during validation.
        sample_model_str (str or List[str], optional): Strings to pass to the model for sampling.
        tokenizer_path (str, optional): Path to tokenizer file for sampling.
        save_checkpoint_steps (int, optional): Interval (in steps) at which to save checkpoints.
        save_checkpoint_path (str, optional): Directory in which to save checkpoint files.
        _model_key (str, optional): Key name for model state in checkpoint dict.
        _optim_key (str, optional): Key name for optimizer state in checkpoint dict.
        _scheduler_key (str, optional): Key name for scheduler state in checkpoint dict.
        wandb_ (bool, optional): Whether to log metrics to Weights & Biases.
        pad_token_id (int, optional): Token ID to ignore in loss calculation.
        val_dataloader_func_config (dict, optional): Config for creating the validation DataLoader.
        generate_val_config (dict, optional): Config for generating samples during validation.
        **kwargs: Additional keyword arguments.

    Raises:
        AssertionError: If save_checkpoint_steps is set without a save path.
        ValueError: If distributed training is enabled but sampler is not DistributedSampler.
        ValueError: If gradient clipping is enabled without specifying max_grad_norm.

    Returns:
        None
    """
    
    warnings.filterwarnings(
        'ignore',
        message=r".*an autograd kernel was not registered to the Autograd key.*"
    ) 
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if save_checkpoint_steps is not None:
        assert save_checkpoint_path, '…path must be set if save_checkpoint_steps is given'

    if parallelism_type in ['fsdp', 'ddp'] and not isinstance(dataloader.sampler, DistributedSampler):
        raise ValueError("For distributed training, dataloader must use DistributedSampler")
    if (checkpoint_batch_idx != 0 or checkpoint_epoch != 0) and isinstance(dataloader.sampler, RandomSampler):
        warnings.warn(f'Resuming at epoch {checkpoint_epoch}, step {checkpoint_batch_idx} with RandomSampler.')
        time.sleep(3)
    if clip_grad_norm and not max_grad_norm:
        raise ValueError('max_grad_norm must be set if grad_norm is True')
    if mixed_precision:
        scaler = GradScaler()
        autocast_dtype = autocast_dtype or torch.float16

    tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_path) if sample_model_val else None
    assert generate_val_config is not None if sample_model_val else True, 'generate_val_config must be set if sample_model_val is True'

    steps = global_steps
    model.train()

    if rank == 0 or parallelism_type not in ['fsdp', 'ddp']:
        print(f"Starting training at epoch {checkpoint_epoch}, step {checkpoint_batch_idx}")

    try:
        for epoch in range(checkpoint_epoch, total_epochs):
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs}",
                    disable=(rank != 0 and parallelism_type in ['fsdp', 'ddp'])) as pbar:
                try:
                    for i, (X, y) in enumerate(dataloader):
                        if epoch == checkpoint_epoch and i < checkpoint_batch_idx:
                            continue
                        X, y = X.to(device), y.to(device)
                        optim.zero_grad()
                        if mixed_precision:
                            with autocast(device_type='cuda', dtype=autocast_dtype):
                                logits = model(X)
                                logits = logits.view(-1, logits.size(-1))
                                targets = y.view(-1)
                                per_token_loss = criterion(logits, targets)
                                if pad_token_id is not None:
                                    mask = (targets != pad_token_id).float()
                                    local_loss_sum = (per_token_loss * mask).sum()
                                    local_token_count = mask.sum() 
                                    if parallelism_type in ['ddp', 'fsdp']:
                                        dist.all_reduce(local_loss_sum, op = dist.ReduceOp.SUM)
                                        dist.all_reduce(local_token_count, op = dist.ReduceOp.SUM)
                                    loss = local_loss_sum / local_token_count 
                                else:
                                    local_loss_sum = per_token_loss.sum()
                                    local_token_count = per_token_loss.numel()
                                    if parallelism_type in ['ddp', 'fsdp']:
                                        dist.all_reduce(local_loss_sum, op = dist.ReduceOp.SUM) 
                                        dist.all_reduce(local_token_count, op = dist.ReduceOp.SUM)
                                    loss = local_loss_sum / local_token_count
                            scaler.scale(loss).backward()
                            scaler.unscale_(optim)
                            if clip_grad_norm:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            if track_grad_norm:
                                grad_norm_dict = get_grad_norm(model)   
                            scaler.step(optim)
                            scaler.update()
                        else:
                            logits = model(X)
                            logits = logits.view(-1, logits.size(-1))
                            targets = y.view(-1)
                            per_token_loss = criterion(logits, targets)
                            if pad_token_id is not None:
                                mask = (targets != pad_token_id).float()
                                local_loss_sum = (per_token_loss * mask).sum()
                                local_token_count = mask.sum() 
                                if parallelism_type in ['ddp', 'fsdp']:
                                    dist.all_reduce(local_loss_sum, op = dist.ReduceOp.SUM)
                                    dist.all_reduce(local_token_count, op = dist.ReduceOp.SUM)
                                loss = local_loss_sum / local_token_count 
                            else:
                                local_loss_sum = per_token_loss.sum()
                                local_token_count = per_token_loss.numel()
                                if parallelism_type in ['ddp', 'fsdp']:
                                    dist.all_reduce(local_loss_sum, op = dist.ReduceOp.SUM)
                                    dist.all_reduce(local_token_count, op = dist.ReduceOp.SUM)
                                loss = local_loss_sum / local_token_count
                            loss.backward()
                            if clip_grad_norm:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            if track_grad_norm:
                                grad_norm_dict = get_grad_norm(model)                   
                            optim.step() 
                        _clear_cache(clr_cuda_cache = True, gc_ = True, X=X, y=y)
                        scheduler.step()
                        steps += 1
                        lr = scheduler.get_last_lr()[0]
                        
                        pplx = _get_pplx(loss.item()) 
                        pbar.set_description(f"Epochs: {epoch + 1}/{total_epochs} | Steps: {steps + 1} | Loss: {loss.item():.4f} | PPLX: {pplx} | LR: {lr}")
                        pbar.update(1)

                        if wandb_ and (rank == 0 or parallelism_type not in ['fsdp', 'ddp']):
                            wandb_dict={"Loss": loss.item(), "Perplexity": pplx, "Learning Rate:": lr}
                            if track_grad_norm:
                                wandb_dict.update(grad_norm_dict) 
                            wandb.log(wandb_dict)

                        if steps % val_steps == 0:
                            model.eval()
                            val_steps = 0
                            val_loss = 0
                            val_pplx = 0
                            with torch.no_grad():
                                val_dataloader = _get_eval_dataloader(**val_dataloader_func_config, parallelism_type = parallelism_type, rank = dist.get_rank())
                                with tqdm(total = len(val_dataloader), desc = f"Validation", disable = (rank != 0 and parallelism_type in ['fsdp', 'ddp'])) as val_pbar:
                                    for i, (X, y) in enumerate(val_dataloader):
                                        val_steps += 1
                                        X, y = X.to(device), y.to(device)
                                        if mixed_precision: 
                                            with autocast(device_type = 'cuda', dtype = autocast_dtype):
                                                logits = model(X)
                                                logits = logits.view(-1, logits.size(-1))
                                                targets = y.view(-1) 
                                                val_per_token_loss = F.cross_entropy(logits, targets, reduction = 'none')
                                                if pad_token_id is not None:
                                                    mask = (targets != pad_token_id).float()
                                                    val_local_loss_sum = (val_per_token_loss * mask).sum()
                                                    val_local_token_count = mask.sum() 
                                                    if parallelism_type in ['ddp', 'fsdp']:
                                                        dist.all_reduce(val_local_loss_sum, op = dist.ReduceOp.SUM)
                                                        dist.all_reduce(val_local_token_count, op = dist.ReduceOp.SUM)
                                                    val_loss += val_local_loss_sum / val_local_token_count 
                                                else:
                                                    val_local_loss_sum = val_per_token_loss.sum()
                                                    val_local_token_count = val_per_token_loss.numel()
                                                    if parallelism_type in ['ddp', 'fsdp']:
                                                        dist.all_reduce(val_local_loss_sum, op = dist.ReduceOp.SUM)
                                                        dist.all_reduce(val_local_token_count, op = dist.ReduceOp.SUM)                                       
                                                    val_loss += val_local_loss_sum / val_local_token_count
                                                val_pplx += _get_pplx(val_loss.item())
                                        else:
                                            logits = model(X)
                                            logits = logits.view(-1, logits.size(-1))
                                            targets = y.view(-1) 
                                            val_per_token_loss = F.cross_entropy(logits, targets, reduction = 'none')
                                            if pad_token_id is not None:
                                                mask = (targets != pad_token_id).float()
                                                val_local_loss_sum = (val_per_token_loss * mask).sum()
                                                val_local_token_count = mask.sum() 
                                                if parallelism_type in ['ddp', 'fsdp']:
                                                    dist.all_reduce(val_local_loss_sum, op = dist.ReduceOp.SUM)
                                                    dist.all_reduce(val_local_token_count, op = dist.ReduceOp.SUM)                                       
                                                val_loss += val_local_loss_sum / val_local_token_count
                                            else:
                                                val_local_loss_sum = val_per_token_loss.sum()
                                                val_local_token_count = val_per_token_loss.numel()
                                                if parallelism_type in ['ddp', 'fsdp']:
                                                    dist.all_reduce(val_local_loss_sum, op = dist.ReduceOp.SUM)
                                                    dist.all_reduce(val_local_token_count, op = dist.ReduceOp.SUM)                                       
                                                val_loss += val_local_loss_sum / val_local_token_count
                                            val_pplx += _get_pplx(val_loss.item())
                                        val_pbar.update(1)
                                val_pplx /= val_steps 
                                val_loss /= val_steps 
                                _clear_cache(clr_cuda_cache = True, gc_ = True, X=X, y=y, targets = targets, mask = mask, logits = logits)
                                
                                if wandb_:
                                    wandb_dict={"Validation Loss": val_loss.item(), "Validation Perplexity": val_pplx}
                                    wandb.log(wandb_dict)
                                print(f'VALIDATION | Loss: {val_loss.item()} | PPLX: {val_pplx}')                               
                                
                            model.train()

                            if steps % save_checkpoint_steps == 0:
                                torch.cuda.synchronize()
                                _call_dist_barrier(parallelism_type)
                                model_state_dict = None
                                optim_state_dict = None
                                if parallelism_type == 'fsdp':
                                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                                        print(f'[Rank 0] Getting FSDP model state dict')
                                        model_state_dict = model.state_dict()
                                        
                                        if optim is not None:
                                            if rank == 0:
                                                print(f"[Rank 0] Getting FSDP optimizer state dict")
                                            optim_state_dict = FSDP.full_optim_state_dict(model, optim)
                                else:
                                    if rank == 0:
                                        model_state_dict = (
                                            model.module.state_dict() if parallelism_type in ['dp', 'ddp']
                                            else model.state_dict()
                                        )
                                        if optim is not None:
                                            optim_state_dict = optim.state_dict()
                                if rank == 0:
                                    print(f'[Rank 0] Saving checkpoint at step {steps}')
                                    save_checkpoint(
                                        model_state_dict=model_state_dict,  
                                        optim_state_dict=optim_state_dict,
                                        scheduler=scheduler,
                                        loss=loss,
                                        pplx=pplx,
                                        epoch=epoch,
                                        batch_idx=i,
                                        rank=rank,
                                        global_steps=steps,
                                        save_checkpoint_path=save_checkpoint_path,
                                        _model_key=_model_key,
                                        _optim_key=_optim_key,
                                        _scheduler_key=_scheduler_key,
                                        parallelism_type=parallelism_type,
                                    ) 
                                print(f'Saved Checkpoint, now exiting')
                                _clear_cache(model_state_dict = model_state_dict, gc_ = False, optim_state_dict = optim_state_dict)
                except RuntimeError as e:
                    if "out of memory in" in str(e):
                        print('OOM, clearing VRAM cache')
                        _clear_cache(clr_cuda_cache = True)
                        continue 
                    else:
                        raise e
                
        print(f"Training completed over {total_epochs} epochs, {steps} steps")

    except KeyboardInterrupt:
        if is_main_process():
            print("KeyboardInterrupt - Stopping training safely.")
    finally:
        model_state_dict = model.module.state_dict() if parallelism_type in ['dp', 'ddp'] else model.state_dict()
        optim_state_dict = optim.state_dict() if optim is not None else None
        save_checkpoint(
            model_state_dict=model_state_dict,  
            optim_state_dict=optim_state_dict,
            scheduler=scheduler,
            loss=loss,
            pplx=pplx,
            epoch=epoch,
            batch_idx=i,
            rank=rank,
            global_steps=steps,
            save_checkpoint_path=save_checkpoint_path,
            _model_key=_model_key,
            _optim_key=_optim_key,
            _scheduler_key=_scheduler_key,
            parallelism_type=parallelism_type,
        ) 
        print(f"Saved Checkpoint, now exiting.")
        if dist.is_initialized():
            _call_dist_barrier(parallelism_type)
            dist.destroy_process_group()

def generate(
    str_in:str, 
    tokenizer, 
    model, 
    eos_token, 
    context_len:int,
    max_toks_out = None, 
    _greedy:bool = False, 
    top_p:float = .5, 
    top_k:int = None, 
    temperature:float = 1.0,
    verbose:bool = False,
    new_line:bool = False,
    *args,
    **kwargs
    ):
 
    """
    Generate text from a language model given an input string.

    Parameters
    ----------
    str_in : str
        Input string to condition the generation.
    tokenizer : object
        Tokenizer to convert between strings and tokens.
    model : torch.nn.Module
        The language model used for text generation.
    eos_token : int
        End-of-sequence token ID to signal when to stop generation.
    context_len : int
        Maximum length of the input context for the model.
    max_toks_out : int, optional
        Maximum number of tokens to generate (default is unlimited).
    _greedy : bool, default=False
        If True, performs greedy decoding (argmax at each step).
    top_p : float, default=0.5
        Top-p (nucleus) sampling threshold for probabilistic decoding.
    top_k : int, optional
        Top-k sampling threshold for probabilistic decoding.
    temperature : float, default=1.0
        Sampling temperature; higher values make output more random.
    verbose : bool, default=False
        If True, prints generation speed and token count.
    new_line : bool, default=False
        If True, adds a newline before and after generation.
    *args, **kwargs :
        Additional arguments passed to the sampling function.

    Returns
    -------
    None
        The generated text is printed to stdout in real-time.
    """
  
    model.eval()
   
    with torch.no_grad():
    
            if new_line:
                print()
            sampler = sample_model(
                str_in = str_in,
                tokenizer = tokenizer,
                model = model,
                context_len = context_len,
                eos_token=eos_token,
                max_toks_out = max_toks_out,
                _greedy = _greedy,
                top_p = top_p,
                top_k = top_k,
                temperature = temperature
            ) 
            n_toks = 0
            if verbose:
                start_time = time.time()
            print(str_in, end = '', flush = True)
            for i in sampler:
                n_toks += 1 
                print(i, end = '', flush = True)
            if verbose: 
                end_time = time.time()
                print(f'\n\nGenerated a total of {n_toks} tokens in {end_time - start_time} seconds')
            if new_line:
                print()

            model.t = None
            
            for block in model.transformer_blocks:
                block.MHSA.K_cache = None
                block.MHSA.V_cache = None

def sample_model(
    str_in: str,
    tokenizer,
    model: nn.Module,
    context_len: int,
    eos_token: int,
    max_toks_out: int = None,
    _greedy: bool = False,
    top_p: float = 0.9,  
    top_k: int = None,
    temperature: float = 0.7 
):
    assert len(str_in) > 0, 'str_in must be non-empty'
    
    _first = True
    _inference = True
    n_tok = 0
   
    x_in = torch.tensor(tokenizer.encode(str_in), dtype=torch.long).unsqueeze(0)
    if x_in.size(-1) > (context_len - 1):
        raise ValueError(f'Input length must be <= {context_len - 1}')
    
    while True:
        x = model(x_in, _inference=_inference, _first=_first)
        if _first:
            _first = False
            x = x[:, -1, :].unsqueeze(1)
        next_tok = _decode_output(x, _greedy=_greedy, top_p=top_p, top_k=top_k, temperature=temperature)
        next_tok_out = next_tok.item()
        if next_tok_out == eos_token or (max_toks_out and n_tok >= max_toks_out):
            break
        yield tokenizer.decode([next_tok_out])
        
        if _greedy:
            x_in = torch.cat([x_in, next_tok.unsqueeze(0).unsqueeze(0)], dim=1)
        else: 
            x_in = torch.cat([x_in, next_tok.unsqueeze(0)], dim=1)
        x_in = x_in[:, -context_len:] if x_in.size(1) > (context_len - 1) else x_in
        n_tok += 1

def _decode_output(x: torch.Tensor, _greedy: bool = False, top_p: float = 0.9, top_k: int = None, temperature: float = 0.7):
    if _greedy:
        return torch.argmax(x, dim=-1).squeeze()
    
    if top_k:
        values, x_idxs = torch.topk(x, k=top_k)
        x_inf_val = torch.full_like(x, float('-inf'))
        x = x_inf_val.scatter(dim=-1, index=x_idxs, src=x.gather(dim=-1, index=x_idxs))
    
    probs = F.softmax(x / temperature, dim=-1)
    
    if top_p:
        x_sorted, x_idxs = torch.sort(probs, descending=True)
        x_sorted_cumsum = torch.cumsum(x_sorted, dim=-1)
        keep_mask = x_sorted_cumsum <= top_p
        keep_mask[..., 0] = True  # Ensure at least one token
        final_mask = torch.zeros_like(probs, dtype=torch.bool)
        final_mask.scatter_(dim=-1, index=x_idxs, src=keep_mask)
        probs = probs * final_mask
        masked_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / masked_sum if masked_sum > 0 else probs
    
    probs = probs.squeeze()
    assert probs.dim() == 1, f"Expected 1D probs, got shape {probs.shape}"
    return torch.multinomial(probs, num_samples=1)

def get_custom_scheduler(optimizer, warmup_steps, constant_steps, decay_steps, max_lr, min_lr):
    """
    Returns a custom learning rate scheduler with warmup, flat, and cosine decay phases.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scheduled.
    warmup_steps : int
        Number of steps to linearly increase learning rate from `min_lr` to `max_lr`.
    constant_steps : int
        Number of steps to keep learning rate constant at `max_lr` after warmup.
    decay_steps : int
        Number of steps over which to apply cosine decay from `max_lr` to `min_lr`.
    max_lr : float
        The maximum learning rate to reach after warmup.
    min_lr : float
        The minimum learning rate to start at and decay to.
    
    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        A PyTorch learning rate scheduler that applies the custom schedule.

    Notes
    -----
    The returned scheduler expects `optimizer.param_groups[0]['initial_lr']` to be set.
    Make sure to define this before creating the scheduler if not using default PyTorch behavior.
    """
    
    def lr_lambda(step):
        if step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (step / warmup_steps)
        elif step < warmup_steps + constant_steps:
            lr = max_lr
        else:
            decay_step = step - (warmup_steps + constant_steps)
            cosine_progress = min(1.0, decay_step / decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(cosine_progress * math.pi))
            lr = min_lr + (max_lr - min_lr) * cosine_decay
        return lr / optimizer.param_groups[0]['initial_lr']
    return LambdaLR(optimizer, lr_lambda)

def get_config(type_: str):
    """
    Loads a JSON configuration file based on the specified type.

    Parameters
    ----------
    type_ : str
        The type of configuration to load. Must be one of:
        ['opt', 'model', 'lr', 'loss', 'train', 'dataloader', 'parallel', 'wandb', 'val_dataloader', 'generate_val'].

    Returns
    -------
    dict
        The parsed configuration dictionary.

    Raises
    ------
    AssertionError
        If `type_` is not one of the valid configuration types.
    FileNotFoundError
        If the corresponding configuration file does not exist.
    JSONDecodeError
        If the JSON file is malformed.
    """
    valid_types = ['opt', 'model', 'lr', 'loss', 'train', 'dataloader', 'parallel', 'wandb', 'val_dataloader', 'generate_val']
    assert type_ in valid_types, f"type_ must be one of {', '.join(valid_types)}."
    config_path = f'src/config/{type_}_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def load_tokenizer(tok_path):
    """
    Loads a tokenizer from a specified tokenizer file path.

    Parameters
    ----------
    tok_path : str
        Path to the tokenizer JSON file.

    Returns
    -------
    PreTrainedTokenizerFast
        A Hugging Face `PreTrainedTokenizerFast` instance initialized from the file.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = tok_path)
    return tokenizer

def load_checkpoint(
    model,
    optim,
    scheduler,
    load_checkpoint_path,
    _model_key,
    _optim_key,
    _scheduler_key,
    parallelism_type,
    model_only:bool = False
    ):

    """
    Loads a training checkpoint and restores the model, optimizer, and scheduler states.

    Parameters
    ----------
    model : torch.nn.Module
        The model instance to load weights into.
    optim : torch.optim.Optimizer
        The optimizer instance to restore state.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to restore state.
    load_checkpoint_path : str
        Path to the checkpoint file.
    _model_key : str
        Key to access the model state dict in the checkpoint.
    _optim_key : str
        Key to access the optimizer state dict in the checkpoint.
    _scheduler_key : str
        Key to access the scheduler state dict in the checkpoint.
    parallelism_type : str
        Type of parallelism used ('fsdp', 'dp', 'ddp', or 'none').
    model_only : bool, optional
        If True, only the model weights are loaded (default: False).

    Returns
    -------
    If model_only is False:
        tuple : (model, optim, epoch, global_steps, loss, batch_idx)
    Else:
        model : torch.nn.Module
    """    
    
    print(f"Loading training checkpoint from {load_checkpoint_path}")
    checkpoint = torch.load(load_checkpoint_path, map_location='cpu')
    if model_only: 
        model.load_state_dict(checkpoint[_model_key])
        return model
    if parallelism_type == 'fsdp':
        model.load_state_dict(checkpoint[_model_key])
        full_optim_state_dict = checkpoint[_optim_key]
        sharded_optim_state_dict = FSDP.shard_full_optim_state_dict(full_optim_state_dict, model)
        optim.load_state_dict(sharded_optim_state_dict)
    else:
        if parallelism_type in ['dp', 'ddp'] and hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint[_model_key])
        else:
            model.load_state_dict(checkpoint[_model_key])
        optim.load_state_dict(checkpoint[_optim_key])
    scheduler.load_state_dict(checkpoint[_scheduler_key])
    epoch = checkpoint['epoch']
    global_steps = checkpoint['global_steps']
    loss = checkpoint.get('loss', None)
    batch_idx = checkpoint.get('batch_idx', 0)
    return model, optim, epoch, global_steps, loss, batch_idx

def setup_parallel(parallelism_type, backend=None, sleep=3, verbose: bool = False):
    """
    Initializes distributed training environment based on the specified parallelism type.

    Parameters
    ----------
    parallelism_type : str
        Type of parallelism to use: 'fsdp', 'ddp', 'dp', or None.
    backend : str, optional
        Backend for distributed training ('nccl' or 'gloo'). Required for 'fsdp' or 'ddp'.
    sleep : int, optional
        Time (in seconds) to wait if CUDA is unavailable (default: 3).
    verbose : bool, optional
        If True, prints status messages during setup (default: False).

    Returns
    -------
    dict
        Dictionary containing:
            - 'device': torch.device to be used,
            - 'world_size': number of processes (None if not distributed),
            - 'rank': global rank (None if not distributed),
            - 'local_rank': local rank (only if using 'fsdp' or 'ddp').
    """
    _par_check = ['fsdp', 'ddp']
    _backend_check = ['nccl', 'gloo']
    if verbose and parallelism_type in _par_check:
        print(f'Setting up {parallelism_type} using the {backend} backend')
    
    rank = None
    local_rank = None
    world_size = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if parallelism_type in _par_check:
        if backend is None:
            raise ValueError('backend must be specified for fsdp or ddp')
        if backend not in _backend_check:
            raise ValueError(f"backend must be one of {', '.join(_backend_check)}")
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        device = _set_device(local_rank, parallelism_type)
    elif parallelism_type in ['dp', None]:
        if device.type == 'cpu':
            warnings.warn('CUDA unavailable, using CPU')
            time.sleep(sleep)
    out = {'device': device, 'world_size': world_size, 'rank': rank}
    if parallelism_type in _par_check:
        out['local_rank'] = local_rank
    return out

def get_model_parallel(
    module,
    device='cuda',
    parallelism_type=None,
    type_=None,
    min_num_params=None,
    backend=None
    ):
    
    """
    Wraps a model with the specified parallelism strategy.

    Parameters
    ----------
    module : nn.Module
        The model to wrap.
    device : str, optional
        Target device to place the model on (default: 'cuda').
    parallelism_type : str, optional
        One of 'dp', 'ddp', 'fsdp', or None. Determines the parallelism strategy.
    type_ : str, optional
        FSDP policy type (required if parallelism_type is 'fsdp').
    min_num_params : int, optional
        Minimum number of parameters to trigger FSDP wrapping (required for 'fsdp').
    backend : str, optional
        Backend used for distributed setup (e.g., 'nccl').

    Returns
    -------
    nn.Module
        Model wrapped in the specified parallelism wrapper, placed on appropriate device.
    """
    
    _parallel_type_list = ['fsdp', 'ddp', 'dp', None]
    if parallelism_type in _parallel_type_list:
        print(f"Preparing model using {parallelism_type}{' and ' + backend if backend else ''}")
    else:
        raise ValueError("parallelism_type must be 'dp', 'ddp', 'fsdp', or None")

    if parallelism_type == 'dp':
        local_rank = 0
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        module = module.to(device)
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(module, device_ids=device_ids)
    elif parallelism_type == 'ddp':
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = f'cuda:{local_rank}'
        module = module.to(device)
        model = DDP(module, device_ids=[local_rank])
    elif parallelism_type == 'fsdp':
        if type_ is None or min_num_params is None:
            raise ValueError("type_ and min_num_params must be specified for 'fsdp'")
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = f'cuda:{local_rank}'
        module = module.to(device)
        model = _wrap_fsdp(module, type_, min_num_params)
    else:  # - 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = module.to(device)
    return model

def save_checkpoint(
    model_state_dict, 
    optim_state_dict,
    scheduler,
    loss,
    pplx,
    epoch,
    batch_idx,
    rank,
    global_steps,
    save_checkpoint_path,
    _model_key,
    _optim_key,
    _scheduler_key,
    parallelism_type=None
    ):
    
    """
    Saves a checkpoint containing the model state, optimizer state, scheduler state,
    training loss, perplexity, and other relevant training information.

    Args:
        model_state_dict (dict): The state dictionary of the model to be saved.
        optim_state_dict (dict): The state dictionary of the optimizer to be saved.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to be saved.
        loss (torch.Tensor or float): The current loss value.
        pplx (float): The perplexity of the model at the current step.
        epoch (int): The current epoch number.
        batch_idx (int): The index of the current batch in the epoch.
        rank (int): The rank of the current process in a distributed setting.
        global_steps (int): The global number of steps taken during training.
        save_checkpoint_path (str): The path where the checkpoint should be saved.
        _model_key (str): The key under which the model state is saved.
        _optim_key (str): The key under which the optimizer state is saved.
        _scheduler_key (str): The key under which the scheduler state is saved.
        parallelism_type (str, optional): The parallelism type used for training 
                                          ('ddp', 'fsdp', or None). Default is None.

    Returns:
        None

    Notes:
        - The checkpoint is saved only on rank 0 in a distributed setting (ddp, fsdp).
        - The checkpoint includes the model state, optimizer state, scheduler state, 
          loss, perplexity, epoch, batch index, and global steps.
        - The checkpoint is saved in a directory structure, with the filename containing 
          the epoch, batch index, and global steps.
    """ 
    
    print(f"Saving checkpoint at epoch {epoch} after {global_steps} steps on rank {rank}")
   
    os.makedirs(save_checkpoint_path, exist_ok = True)
    
    checkpoint = {
        _model_key: model_state_dict,
        _optim_key: optim_state_dict,
        _scheduler_key: scheduler.state_dict(),
        'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        'pplx': pplx,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'global_steps': global_steps
    }

    if parallelism_type in ['ddp', 'fsdp'] and rank != 0:
        return

    save_path = os.path.join(save_checkpoint_path, f'checkpoint_epoch_{epoch}_step_{batch_idx}_global_step_{global_steps}')
    print(f'Saving checkpoint to {save_path} on rank {rank}')
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path} on rank {rank}")

def _wrap_fsdp(module, type_: str = 'size', min_num_params: int = None, transformer_layer_cls: nn.Module = None, **fsdp_kwargs):
    """
    Wraps a model in Fully Sharded Data Parallel (FSDP) based on the specified wrapping policy.

    Args:
        module (nn.Module): The module (model) to be wrapped in FSDP.
        type_ (str, optional): The type of auto-wrapping policy to use. 
                               Can be either 'size' (for size-based wrapping) or 'transformer' (for transformer-based wrapping). Default is 'size'.
        min_num_params (int, optional): The minimum number of parameters for a module to be wrapped in FSDP, required if `type_` is 'size'.
        transformer_layer_cls (nn.Module, optional): The class of transformer layer to be used for wrapping in transformer-based FSDP. 
                                                     If not provided, defaults to `TransformerBlock`.
        **fsdp_kwargs: Additional keyword arguments passed to the FSDP wrapper.

    Returns:
        FSDP: The wrapped module with FSDP applied.

    Raises:
        ValueError: If `type_` is not one of 'size' or 'transformer', or if `min_num_params` is not specified when using 'size'.

    Notes:
        - If `type_` is 'size', FSDP will use a size-based policy and will wrap modules that have a parameter count larger than `min_num_params`.
        - If `type_` is 'transformer', FSDP will use a transformer-based wrapping policy based on the provided `transformer_layer_cls`.
    """
    
    _type_list = ['size', 'transformer']
    if type_ not in _type_list:
        raise ValueError("type_ must be either 'size' or 'transformer'")
    if type_ == 'size':
        if min_num_params is None:
            raise ValueError("min_num_params must be specified for 'size'")
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:  # transformer
        if transformer_layer_cls is None:
            transformer_layer_cls = TransformerBlock
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
    return FSDP(module, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)

def _set_device(rank, parallelism_type):
    """
    Sets the device for the given rank in a multi-GPU setup, ensuring that the rank corresponds to an available GPU.

    Args:
        rank (int): The rank of the current process in the distributed training setup.
        parallelism_type (str): The parallelism type used (e.g., 'fsdp', 'ddp'). If CUDA is unavailable, this is used to raise errors appropriately.

    Returns:
        torch.device: The device (CUDA or CPU) assigned to the current rank.

    Raises:
        RuntimeError: If CUDA is unavailable and the parallelism type requires it (e.g., 'fsdp', 'ddp'), or if the `rank` exceeds available GPUs.

    Notes:
        - This function ensures that each rank is assigned a unique GPU if CUDA is available.
        - If CUDA is unavailable, the function will raise an error for parallelism types that require GPUs and fall back to using the CPU otherwise.
    """
    if torch.cuda.is_available():
        if rank >= torch.cuda.device_count():
            raise RuntimeError(f"Rank {rank} exceeds available GPUs ({torch.cuda.device_count()})")
        torch.cuda.set_device(rank)
        return torch.device(f'cuda:{rank}')
    if parallelism_type in ['fsdp', 'ddp']:
        raise RuntimeError(f"CUDA unavailable for {parallelism_type}")
    warnings.warn('Using CPU')
    return torch.device('cpu')

def init_wandb(**wandb_kwargs):
    """
    Initializes a Weights and Biases (WandB) session with the provided configuration.

    This function sets the WandB API key, configures the session settings, and starts a new WandB run. 
    It also sets environment variables for WandB-related settings.

    Args:
        **wandb_kwargs: A dictionary of keyword arguments that can include:
            - 'api_key' (str): The Weights and Biases API key (required).
            - 'project' (str): The name of the WandB project.
            - 'dir' (str): The directory for saving WandB logs and data.
            - 'name' (str): The name of the current run.
            - 'id' (str): The run ID to be used for this session.

    Raises:
        AssertionError: If the 'api_key' is not provided in the keyword arguments.
    
    Notes:
        - This function requires that the WandB API key is provided in `wandb_kwargs['api_key']`.
        - The environment variable `WANDB_API_KEY` is set automatically.
        - The `WANDB_INIT_TIMEOUT` is set to 600 seconds by default.
        - Debugging options for WandB can be enabled by uncommenting the relevant lines in the code.
    """
    assert wandb_kwargs.get('api_key'), "WandB API key must be provided"
    os.environ['WANDB_API_KEY'] = wandb_kwargs['api_key']
    os.environ['WANDB_INIT_TIMEOUT'] = '600'
#    os.environ['WANDB_DEBUG']=True
#    os.environ['WANDB_CORE_DEBUG']=True
    
    wandb.login()
    wandb.init(
        project = wandb_kwargs.get('project') ,
        dir = wandb_kwargs.get('dir'),
        name = wandb_kwargs.get('name'),
        id = wandb_kwargs.get('id')
    )

def get_grad_norm(model):
    """
    Computes the L2 norm of gradients for each parameter in the model.

    This function iterates over all parameters in the model and computes the L2 norm (Euclidean norm) 
    of the gradients for each parameter. The gradient norms are returned in a dictionary where keys are 
    the parameter names and values are the corresponding L2 norms.

    Args:
        model (torch.nn.Module): The model whose gradients' norms are to be computed. 

    Returns:
        dict: A dictionary where the keys are the names of the parameters (strings) and the values are 
              the L2 norms of the gradients (floats). If a parameter does not have a gradient, it is skipped.
    
    Notes:
        - The function uses the `.named_parameters()` method to access each parameter's name and value.
        - If the gradient of a parameter is `None`, it is skipped and not included in the output.
        - The names of the parameters are cleaned by removing certain substrings (`'_fsdp_wrapped_module.'` 
          and `'.'_flat_param'`), which might be used in certain parallelization strategies (e.g., FSDP).
        - The function assumes the gradients have already been computed and stored in the model parameters.
    """
    grad_norm_dict = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            clean_name = name.replace('_fsdp_wrapped_module.', '').replace('._flat_param', '')
            param_norm = p.grad.norm(2)
            grad_norm_dict[clean_name] = param_norm
    return grad_norm_dict
   
def _save_config(input_dir, run, output_dir):
    """
    Saves the configuration files from the input directory into a single text file.

    This function iterates through the files in the `input_dir`, looking for JSON files. It attempts to read 
    each JSON file and writes its content into a single text file located in `output_dir`. The output file is 
    organized by run number, and each file's content is prefixed with its filename for easy reference.

    Args:
        input_dir (str): The directory containing the JSON configuration files to be saved.
        run (int): The run number, which is used to create a subdirectory in `output_dir` for saving the file.
        output_dir (str): The directory where the configuration data will be saved.

    Returns:
        None: The function does not return any values. It writes the configuration data to a text file in the output directory.

    Notes:
        - The output file is named `configs.txt` and is stored in a subdirectory named `run_<run>`, where `<run>` is the run number.
        - Each JSON file's contents are written with a `--- <filename> ---` header, followed by the pretty-printed JSON data.
        - If any errors occur while reading a JSON file (e.g., invalid JSON or file reading errors), a warning is printed, and the file is skipped.
        - The function handles `json.JSONDecodeError` and other generic exceptions when reading the JSON files.
        - If the output directory or subdirectory does not exist, they will be created.
    """
    run_subdir = os.path.join(output_dir, f"run_{run}")
    os.makedirs(run_subdir, exist_ok=True)
    
    output_file = os.path.join(run_subdir, 'configs.txt')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(input_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(input_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            json_data = json.load(infile)
                        outfile.write(f"--- {filename} ---\n")
                        outfile.write(json.dumps(json_data, indent=2))
                        outfile.write("\n\n")
                    except json.JSONDecodeError:
                        print(f"Warning: '{filename}' contains invalid JSON. Skipping.")
                    except Exception as e:
                        print(f"Warning: Error reading '{filename}': {e}. Skipping.")
        print(f"Successfully wrote data to '{output_file}'.")
    except Exception as e:
        print(f"Error: Failed to write to '{output_file}': {e}.")
       
def _get_eval_dataloader(
    eval_data_path, 
    eval_size, 
    val_batch_size, 
    num_workers, 
    shuffle, 
    pin_memory, 
    parallelism_type, 
    rank,
    **kwargs
    ):

    """
    Creates a DataLoader for evaluation data.

    This function loads the evaluation data (`X_val` and `Y_val`) from the specified path, selects a subset of the 
    data based on `eval_size`, and then creates a DataLoader for it. The DataLoader is configured to handle 
    distributed or parallel processing based on the provided `parallelism_type` and `rank`. This function is 
    useful for preparing evaluation data during training for tasks like validation or testing.

    Args:
        eval_data_path (str): Path to the directory containing the evaluation data files. 
                               It should contain 'X_val.pt' and 'Y_val.pt' for features and labels respectively.
        eval_size (int): The number of samples to be used for evaluation. This limits the size of the evaluation data.
        val_batch_size (int): The batch size to be used when loading the evaluation data.
        num_workers (int): The number of subprocesses to use for data loading. 
        shuffle (bool): Whether to shuffle the evaluation data during loading.
        pin_memory (bool): Whether to pin memory for faster data transfer to the GPU.
        parallelism_type (str): The type of parallelism to be used. Options include 'dp', 'ddp', 'fsdp', or None.
        rank (int): The rank of the current process in distributed training (useful for distributed settings).
        **kwargs: Additional keyword arguments passed to the `get_dataloader` function for further customization.

    Returns:
        DataLoader: A DataLoader object configured to load the evaluation data with the specified settings.

    Raises:
        AssertionError: If the shapes of `X_val` and `y_val` do not match.
    """

    X_val_path = os.path.join(eval_data_path, 'X_val.pt')
    Y_val_path = os.path.join(eval_data_path, 'Y_val.pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X = torch.load(X_val_path, map_location = 'cpu')
        y = torch.load(Y_val_path, map_location = 'cpu')
    X_val = X[:eval_size, :512] 
    y_val = y[:eval_size, :512] 
    del X, y
    assert X_val.shape == y_val.shape, 'X_val and y_val must have the same shape' 
    val_dataloader = get_dataloader(
        X = X_val, 
        y = y_val, 
        batch_size = val_batch_size, 
        num_workers = num_workers, 
        shuffle = shuffle, 
        pin_memory = pin_memory, 
        parallelism_type = parallelism_type, 
        rank = rank
        )
    return val_dataloader
        
def _clear_cache(clr_cuda_cache: bool = False, gc_ = False, *args, **kwargs):
    """
    Clears the memory cache, deletes specified tensors, and optionally runs garbage collection.

    Args:
        clr_cuda_cache (bool, optional): If True, clears the CUDA memory cache. Defaults to False.
        gc_ (bool, optional): If True, runs garbage collection. Defaults to False.
        *args: Variable number of tensor arguments. Each tensor will be deleted.
        **kwargs: Additional keyword arguments (not used, but can be extended).
    
    Notes:
        - If `clr_cuda_cache` is True, it clears the CUDA memory cache using `torch.cuda.empty_cache()`.
        - If `gc_` is True, it runs garbage collection using `gc.collect()`.
    """
    for i in args:
        if isinstance(i, torch.Tensor): 
            del i 
    if clr_cuda_cache:
        torch.cuda.empty_cache()
    if gc_:
        gc.collect()

def is_main_process():
    """
    Determines if the current process is the main process in a distributed setting.

    Returns:
        bool: True if the current process is the main process, False otherwise. 
              If distributed training is not initialized, returns True.
    """
    return dist.get_rank() == 0 if dist.is_initialized() else True
        
def __exit():
    """
    Cleans up and exits the program by destroying the distributed process group and terminating the program.

    This should be called when the program needs to exit gracefully in a distributed setup.
    """
    dist.destroy_process_group()
    sys.exit(0)
    
def _call_dist_barrier(parallelism_type):
    """
    Synchronizes all processes in a distributed setting by calling a barrier.

    Args:
        parallelism_type (str): The type of parallelism used ('ddp', 'fsdp', etc.).
    
    Notes:
        - If `parallelism_type` is 'ddp' or 'fsdp', it calls `dist.barrier()` to synchronize all processes.
        - If not using distributed parallelism, the function does nothing.
    """
    if parallelism_type in ['ddp', 'fsdp']:
        dist.barrier()
    else:
        pass
    
def _get_pplx(loss: float):
    """
    Calculates the perplexity from a given loss value.

    Args:
        loss (float): The loss value.

    Returns:
        float: The calculated perplexity. If the calculation causes an overflow, returns `float('inf')`.
    
    Notes:
        - Perplexity is calculated as the exponent of the loss (`math.exp(loss)`).
        - If the loss value leads to an overflow, the function returns infinity (`float('inf')`).
    """
    try:
        pplx = math.exp(loss)
    except OverflowError:
        pplx = float('inf') 
    return pplx
