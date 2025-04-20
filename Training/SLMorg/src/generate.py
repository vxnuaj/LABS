import torch
from transformers import PreTrainedTokenizerFast
from slm import SLM
from utils import generate, get_config

prompt = '''
Once upon a time there was a little girl named Lucy. She was very adventurous. She loved to explore the
world around her, especially when it was bright and sunny outside.
One day, while exploring the nearby park, Lucy came across a ladder leaning on a wall. She was curious
to see what’s on top, so she climbed the ladder, but when she reached the top, the ladder fell and she was
stuck.

'''

tokenizer_file = 'data/data/tokenizer.json'
eos_token = 0
context_len = 512
max_toks_out = 256
_greedy = False
top_p = .5
top_k = 40
temperature = .5
verbose = True

tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file)
model_config = get_config(type_ = 'model')
model = SLM(**model_config)

weights = torch.load(
    'weights/checkpoint_epoch_6_step_6821_global_step_135000',
    weights_only = True
    )['model_state_dict']

model.load_state_dict(weights)

generate(
    str_in = prompt,
    tokenizer = tokenizer,
    model = model,
    eos_token = eos_token,
    context_len = context_len,
    max_toks_out = max_toks_out,
    _greedy = _greedy,
    top_p = top_p,
    top_k = top_k,
    temperature = temperature,
    verbose = verbose
    )