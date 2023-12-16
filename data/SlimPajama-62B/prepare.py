import struct
import os
import sys
import tiktoken
from datasets import load_dataset # huggingface datasets

if __name__ == '__main__':
    dataset = load_dataset("venketh/SlimPajama-62B", streaming=True)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset['train'].map(process, remove_columns=['text', 'meta'])
    f = open('train.bin', 'ab')
    for split in tokenized:
      f.write(struct.pack('%sH' % len(split['ids']), *split['ids']))
    f.close()

    tokenized = dataset['validation'].map(process, remove_columns=['text', 'meta'])
    f = open('val.bin', 'ab')
    for split in tokenized:
      f.write(struct.pack('%sH' % len(split['ids']), *split['ids']))
    f.close()
