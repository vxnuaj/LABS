import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import data_process, tokenize, train_new_tokenizer_bpe, create_val_sequences, create_train_sequences_gen, download_save_data

save_dir = 'data/data'
file_1 = 'data/data/train1.parquet'
file_2 = 'data/data/train2.parquet'
file_3 = 'data/data/train3.parquet'
file_4 = 'data/data/train4.parquet'
file_val = 'data/data/validation.parquet'

file_train = [file_1, file_2, file_3, file_4]
file_val = [file_val]

# PARAMS -----------------
return_single_str = False
vocab_size = 10_000
special_tokens = {'eos': '<|endoftext|>', 'pad': '<|pad|>'}
save_tokenizer_path = 'data/data/tokenizer.json'
context_len = 512
processes = 24
flat_tensor = True
flat_tensor_val = False
seq_tensor_size = 25_000
val_seq_tensor_size = None
max_toks = 350_000_000  
val_max_toks = None
batch_first = True
val_train_n_samples = 2000

X_train_pth = 'data/tensors/train/X'
y_train_pth = 'data/tensors/train/y'
val_pth = 'data/tensors/val'

if __name__ == "__main__":
   
    download_save_data(save_dir) 
    
    os.makedirs(X_train_pth, exist_ok=True)
    os.makedirs(y_train_pth, exist_ok=True)
    os.makedirs(val_pth, exist_ok=True)
    
    data = data_process(
        files=file_train,
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    tokenizer = train_new_tokenizer_bpe(
        data=data['text'].tolist(),
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        save_path=save_tokenizer_path
    )
    
    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor,
        processes=processes
    )

    if isinstance(seq_tensor_size, int):
        sequence_generator = create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )
        for i, (X, y) in enumerate(sequence_generator):
            torch.save(X, os.path.join(X_train_pth, f'X_train_{i}.pt'))
            torch.save(y, os.path.join(y_train_pth, f'y_train_{i}.pt'))
           
    elif not seq_tensor_size:
        X_train, y_train = create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )
        torch.save(X_train, os.path.join(X_train_pth, "X_train.pt"))
        torch.save(y_train, os.path.join(y_train_pth, "y_train.pt"))
        del X_train, y_train

    data = data_process(
        files=file_val,
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor_val,
        processes=processes
    )

    X_val, Y_val = create_val_sequences(
        data=data_tensor,
        batch_first=batch_first,
        padding_value=tokenizer.encode(special_tokens['pad']).ids[0]
    )
   
    X_val_train = X_val[0:val_train_n_samples+1]
    Y_val_train = Y_val[0:val_train_n_samples+1]
    
    torch.save(X_val_train, os.path.join(val_pth, 'X_val.pt'))
    torch.save(Y_val_train, os.path.join(val_pth, 'Y_val.pt'))