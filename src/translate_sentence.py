import os
import sys  
import torch
import numpy as np
from tqdm.auto import tqdm

def translate_sentence(
    args,
    model,
    accelerator,
    tokeniser,
    txt, 
    src_lang,
    trg_lang
):
    decoder_start_token_id = tokeniser.get_vocab()[trg_lang]
    tokeniser.src_lang = src_lang
    item = tokeniser(txt, add_special_tokens=True)
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model.model).generate(
            torch.as_tensor([item['input_ids']]).cuda(), 
            attention_mask=torch.as_tensor([item['attention_mask']]).cuda(), 
            num_beams=args.num_beams,
            decoder_start_token_id=decoder_start_token_id
        ) 
    translation = tokeniser.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return translation
