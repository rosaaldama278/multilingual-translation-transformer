'''
Author: Ting Aldama tingaldama278@gmail.com
Date: 2024-05-22 08:06:01
LastEditors: Ting Aldama tingaldama278@gmail.com
LastEditTime: 2024-07-29 12:41:01
FilePath: /multi-lang-translation/scripts/inference.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
import numpy as np
import torch
import sentencepiece as spm
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer import EncoderDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(src_lang, tgt_lang):
    num_layers = 4
    num_heads = 4
    d_model = 256
    d_ff = 1024

    encoder = Encoder(vocab_size=10000, n_layer=num_layers, n_head=num_heads, d_model=d_model, d_ff=d_ff)
    decoder = Decoder(vocab_size=10000, n_layer=num_layers, n_head=num_heads, d_model=d_model, d_ff=d_ff)
    model = EncoderDecoder(encoder, decoder, device).to(device)
    model_path = os.path.join('checkpoints', src_lang + '-' + tgt_lang, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    spm_model_path = os.path.join('data', src_lang + '-' + tgt_lang, 'bpe', 'bpe.model')
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)

    return model, sp

def translate(src_sentence, model, sp, max_length=10):
    model.eval()

    src_tokens = sp.encode_as_ids(src_sentence)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        src_mask = model.padding_mask(src_tensor)
        memory = model.encoder(src_tensor, src_mask)

    trg_indexes = [sp.bos_id()]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.target_mask(trg_tensor)

        with torch.no_grad():
            output = model.decoder(trg_tensor, memory, src_mask, trg_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

        if pred_token == sp.eos_id():
            break

    trg_tokens = sp.decode_ids(trg_indexes)
    return trg_tokens
