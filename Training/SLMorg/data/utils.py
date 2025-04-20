import torch
import pandas as pd
import pyarrow as pq
import os
import requests

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from typing import List, Union, Tuple
from tqdm import tqdm

def stream_parquet_texts(file_paths: List[str]):
    """
    Streams text data from multiple Parquet files in batches.

    Reads the "text" column from each Parquet file in `file_paths` in batches of 10,000 rows, yielding each text string
    individually. This streaming approach is ideal for processing large datasets that cannot fit entirely in memory.

    Parameters:
    -----------
    file_paths : List[str]
        List of file paths to Parquet files containing a "text" column.

    Yields:
    -------
    str
        A single text string from the "text" column of the Parquet files.
    """
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=10000):
            df = batch.to_pandas()
            yield from df['text'].tolist()

def data_process(files: list, eos_str: str = None, return_single_str: bool = False, return_list_str: bool = False, processes: int = 0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
    """
    Processes text data from Parquet fi es with options for formatting and parallel reading.

    Reads Parquet files either sequentially or in parallel (based on `processes`), combines them into a DataFrame, and
    optionally appends an end-of-sequence (EOS) string to each text entry. The return format varies based on input flags.

    Parameters:
    -----------
    files : list
        List of file paths to Parquet files.
    eos_str : str, optional (default=None)
        String to append to each sequence in the "text" column if provided.
    return_single_str : bool, optional (default=False)
        If True, returns the DataFrame and all text concatenated into a single string.
    return_list_str : bool, optional (default=False)
        If True, returns the "text" column as a list of strings.
    processes : int, optional (default=0)
        Number of processes for parallel file reading. If 0, reads files sequentially.

    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]
        - If `return_list_str` is True: List of text strings.
        - If `return_single_str` is True: Tuple of (DataFrame, concatenated string).
        - Otherwise: Processed DataFrame with a "text" column.
    """
    tqdm.pandas()
    if processes > 0:
        with Pool(processes=processes) as pool:
            dfs = list(tqdm(pool.map(pd.read_parquet, files), total=len(files), desc="Reading Files"))
    else:
        dfs = [pd.read_parquet(f_path) for f_path in tqdm(files, desc="Reading Files")]
    data = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if eos_str:
        print("Adding EOS string to every sequence")
        data['text'] = data['text'] + eos_str
    if return_list_str:
        print(f"Returning list of {len(data)} strings")
        return data['text'].tolist()
    if return_single_str:
        print(f"Concatenating {len(data)} sequences into a single string and returning")
        return data, "".join(data['text'])
    return data

def train_new_tokenizer_bpe(data: List[str], vocab_size, special_tokens: list, save_path=None) -> Tokenizer:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer on a list of strings.

    Initializes a BPE tokenizer with ByteLevel pre-tokenization and decoding, trains it on the input data with a specified
    vocabulary size and special tokens, and optionally saves it to a file.

    Parameters:
    -----------
    data : List[str]
        List of text strings to train the tokenizer on.
    vocab_size : int
        Target vocabulary size for the tokenizer.
    special_tokens : list
        List of special tokens to include in the tokenizer's vocabulary.
    save_path : str, optional (default=None)
        Path to save the trained tokenizer if provided.

    Returns:
    --------
    Tokenizer
        The trained BPE tokenizer object.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
    tokenizer.train_from_iterator(data, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer

def tokenize_chunk(chunk_data, tokenizer):
    """
    Tokenizes a chunk of text data using a provided tokenizer.

    Helper function that encodes a list of strings into token IDs using the given tokenizer, returning a list of token ID lists.

    Parameters:
    -----------
    chunk_data : list
        List of strings to tokenize.
    tokenizer : Tokenizer
        Tokenizer object used to encode the strings.

    Returns:
    --------
    list
        List of lists, where each sublist contains token IDs for a corresponding string.
    """
    encodings = tokenizer.encode_batch(chunk_data)
    token_ids = [enc.ids for enc in encodings]
    return token_ids

def tokenize(data: pd.DataFrame, tokenizer, flat_tensor: bool = True, processes: int = 1) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Tokenizes text data from a DataFrame using parallel processing.

    Splits the DataFrame's "text" column into chunks, tokenizes each chunk in parallel using the specified number of processes,
    and returns either a flattened tensor or a list of tensors based on `flat_tensor`.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with a "text" column containing strings to tokenize.
    tokenizer : Tokenizer
        Tokenizer object for encoding the text.
    flat_tensor : bool, optional (default=True)
        If True, returns a single flattened 1D tensor of all token IDs. If False, returns a list of tensors per sequence.
    processes : int, optional (default=1)
        Number of processes for parallel tokenization.

    Returns:
    --------
    Union[torch.Tensor, List[torch.Tensor]]
        - If `flat_tensor` is True: 1D tensor of all token IDs.
        - If `flat_tensor` is False: List of 1D tensors, each representing a sequence's token IDs.
    """
    print(f"Tokenizing {len(data)} strings with {processes} processes")
    chunk_size = max(1, len(data) // processes)
    chunks = [data['text'].iloc[i:i + chunk_size].tolist() for i in range(0, len(data), chunk_size)]
    with Pool(processes=processes) as pool:
        tokenize_partial = partial(tokenize_chunk, tokenizer=tokenizer)
        results = list(tqdm(pool.map(tokenize_partial, chunks), total=len(chunks), desc="Tokenizing in parallel"))
    token_ids = [ids for chunk in results for ids in chunk]
    if flat_tensor:
        total_tokens = sum(len(ids) for ids in token_ids)
        data_flat = torch.zeros(total_tokens, dtype=torch.long)
        offset = 0
        for ids in tqdm(token_ids, desc="Processing Tensors"):
            data_flat[offset:offset + len(ids)] = torch.tensor(ids, dtype=torch.long)
            offset += len(ids)
        return data_flat
    return [torch.tensor(ids) for ids in token_ids]

def generate_sequence_batch(data, start_indices, context_len):
    """
    Generates a batch of training sequences from a data tensor.

    Extracts sequences of length `context_len` from the data tensor at specified start indices, creating input (X) and target (y)
    sequences shifted by one token.

    Parameters:
    -----------
    data : torch.Tensor
        1D tensor of token IDs.
    start_indices : list
        List of starting indices for sequence extraction.
    context_len : int
        Length of each sequence.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of stacked X (input) and y (target) tensors.
    """
    X_tnsr, y_tnsr = [], []
    for start_idx in start_indices:
        if start_idx + context_len + 1 > len(data):
            break
        X_tnsr.append(data[start_idx:start_idx + context_len])
        y_tnsr.append(data[start_idx + 1:start_idx + context_len + 1])
    return torch.stack(X_tnsr, dim=0), torch.stack(y_tnsr, dim=0)

def create_train_sequences_gen(data: torch.Tensor, context_len: int, seq_tensor_size: int = None, max_toks: int = None, processes: int = 4):
    """
    Generates training sequences from a data tensor, either as a generator or a single tensor pair.

    Creates sequences of length `context_len` with a step size based on `max_toks`. If `seq_tensor_size` is provided, yields
    batches of sequences using parallel threads; otherwise, returns all sequences at once.

    Parameters:
    -----------
    data : torch.Tensor
        1D tensor of token IDs.
    context_len : int
        Length of each sequence.
    seq_tensor_size : int, optional (default=None)
        Number of sequences per batch if yielding batches; if None, generates all sequences at once.
    max_toks : int, optional (default=None)
        Maximum number of tokens to process; must be provided.
    processes : int, optional (default=4)
        Number of threads for parallel batch generation.

    Yields or Returns:
    ------------------
    - If `seq_tensor_size` is int: Yields tuples of (X, y) tensors for each batch.
    - Otherwise: Returns a tuple of (X, y) tensors containing all sequences.
    """
    assert max_toks is not None, 'max_toks must be assigned'
    max_toks = min(max_toks, len(data))
    num_sequences = max_toks // context_len
    step_size = (len(data) - context_len) // num_sequences
    print(f"Creating {num_sequences} sequences with step size {step_size}")
    if isinstance(seq_tensor_size, int):
        batch_size = seq_tensor_size
        total_batches = (num_sequences + batch_size - 1) // batch_size
        start_indices = [i * step_size for i in range(num_sequences)]
        batches = [start_indices[i:i + batch_size] for i in range(0, len(start_indices), batch_size)]
        with ThreadPoolExecutor(max_workers=processes) as executor:
            generate_partial = partial(generate_sequence_batch, data, context_len=context_len)
            for batch_result in tqdm(executor.map(generate_partial, batches), total=len(batches), desc="Generating sequence batches"):
                X, y = batch_result
                yield X, y
    else:
        X_tnsr, y_tnsr = [], []
        for i in tqdm(range(num_sequences), desc=f"Creating {num_sequences} sequences"):
            start_idx = i * step_size
            if start_idx + context_len + 1 > len(data):
                break
            X_tnsr.append(data[start_idx:start_idx + context_len])
            y_tnsr.append(data[start_idx + 1:start_idx + context_len + 1])
        return torch.stack(X_tnsr, dim=0), torch.stack(y_tnsr, dim=0)

def create_val_sequences(data: List[torch.Tensor], batch_first: bool = True, padding_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates padded sequences for validation from a list of tensors.

    Takes a list of variable-length token sequence tensors, generates input (X) and target (y) sequences by shifting, and pads
    them to uniform length for validation purposes.

    Parameters:
    -----------
    data : List[torch.Tensor]
        List of 1D tensors representing token sequences.
    batch_first : bool, optional (default=True)
        If True, returns tensors with shape (batch_size, max_seq_length); if False, (max_seq_length, batch_size).
    padding_value : int, optional (default=0)
        Value used to pad shorter sequences.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of padded X (input) and y (target) tensors.
    """
    X = [seq[:-1] for seq in data]
    y = [seq[1:] for seq in data]
    X_padded = pad_sequence(sequences=X, batch_first=batch_first, padding_value=float(padding_value))
    y_padded = pad_sequence(sequences=y, batch_first=batch_first, padding_value=float(padding_value))
    return X_padded, y_padded

def download_save_data(save_dir: str = 'data/data'):
    os.makedirs(save_dir, exist_ok=True)

    urls = [
        ('train1.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet?download=true'),
        ('train2.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00001-of-00004-5852b56a2bd28fd9.parquet?download=true'),
        ('train3.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00002-of-00004-a26307300439e943.parquet?download=true'),
        ('train4.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00003-of-00004-d243063613e5a057.parquet?download=true'),
        ('validation.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet?download=true'),
    ]

    for new_name, url in tqdm(urls, desc="Downloading"):
        tmp_name = url.split("/")[-1].split("?")[0]
        tmp_path = os.path.join(save_dir, tmp_name)
        final_path = os.path.join(save_dir, new_name)

        response = requests.get(url)
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        os.rename(tmp_path, final_path)
