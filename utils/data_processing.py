import xmltodict 
from gzip import GzipFile
import sentencepiece as spm 
import os 
import json 

def extract_language_pairs(file_path):
    language_pairs = []

    def get_lang_pair(_, tree):
        lang_pair = {}
        for elem in tree['tuv']:
            language = elem['@xml:lang']
            text = elem['seg']
            lang_pair[language] = text
        language_pairs.append(lang_pair)
        return True

    xmltodict.parse(
        GzipFile(file_path),
        item_depth=3, 
        item_callback=get_lang_pair
    )
    return language_pairs

def train_bpe_tokneizer(input_file, model_prefix):
    vocab_size = 10000  # Adjust the vocabulary size as needed
    model_type = 'bpe'  # Use the BPE model type

    # Define the special tokens
    bos_token = '<s>'
    eos_token = '</s>'

    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    shuffle_input_sentence=True,
    input_sentence_size=1000000,
    character_coverage=0.9995,
    bos_piece=bos_token,
    eos_piece=eos_token,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
   )
    

    
def tokenize_sentence(src_input_file,tgt_input_file, model_prefix, ouput_file):
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    src_encoded = []
    tgt_encoded = []
    with open(src_input_file, 'r') as src_f, open(tgt_input_file, 'r') as tgt_f:
        src_lines = src_f.readlines()
        for line in src_lines:
            src_encoded.append(sp.encode_as_ids(line))
        tgt_lines = tgt_f.readlines()
        for line in tgt_lines:
            # add bos and eos to the tgt sentence  
            tgt_encoded.append([sp.bos_id()] + sp.encode_as_ids(line) + [sp.eos_id()])
    file_dir = os.path.dirname(src_input_file)
    with open(os.path.join(file_dir, 'encoded_data.json'), 'w') as f:
        json.dump({"src": src_encoded, "tgt": tgt_encoded })
        
    
