from .simple_tokenizer import SimpleTokenizer


def build_tokenizer(tokenizer_name):
    tokenizer = None
    if tokenizer_name == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in tokenizer_name:
        from .hfpt_tokenizer import HFPTTokenizer

        tokenizer = HFPTTokenizer(pt_name=tokenizer_name[3:])
    elif 'hfc_' in tokenizer_name:
        from .hfpt_tokenizer import HFPTTokenizer
        tokenizer = HFPTTokenizer(pt_name=tokenizer_name[4:])
    else:
        raise ValueError('Unknown tokenizer')

    return tokenizer
