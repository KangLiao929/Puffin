import argparse
import torch
from xtuner.model.utils import guess_load_checkpoint
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False) 
    return original_load(*args, **kwargs)

torch.load = patched_load

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    model = guess_load_checkpoint(args.input)

    torch.save(model, args.output)