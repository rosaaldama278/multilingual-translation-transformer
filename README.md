
# English-Chinese translation app 

This project provides an English to Chinese translation model using a Transformer architecture. The project is organized to be easily downloaded, set up, and used for both inference and further training.

## Features
* Jieba + BPE tokenization for chinese language:
    *  Experimented with multiple tokenization methods, and the combination of pre-tokenization with Jieba and pre trained BPE yielded the best results for the Chinese language
* Trained with well-structured and high-quality English-Chinese parallel corpora



## Setup 

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd en_zh_translation

2. **Install dependencies**: 
   ```bash
   pip install -r resources/requirements.txt


## Inference 
### Run the inference script:
```bash
python scripts/inference.py "This is an example sentence."

example: 
❯ python scripts/inference.py "tell me how you feel"
Translated Sentence: 告訴我你感觉如何?
```
### Inference detail:
* The script will load the models checkpoint from 'checkpoints/model_epoch_11.pt' 
* It will process the input sentence, encode it will trained spm model located under data/spm, and the output the translated chinese sentence 


## Further training 
### Run the training script:
```bash
    python scripts/train.py
```
### Training details: 
* The script will read from data/encoded_data.json
* train_loader and validation_loader will be created using MyDataset and collate_fn from utils/dataset.py
* The model checkpoint will be saved in the checkpoints/ directory



## Model Architecture
The translation model utilizes a Transformer architecture, which consists of an encoder and a decoder based on this famous paper: Attention Is All You Need. Different model settings were experimented and made the model slightly less complex compared to the original paper. The encoder processes the input English sentence, and the decoder generates the corresponding Chinese translation.


## Dataset
The dataset was obtained from this open-source nlp data: https://opus.nlpl.eu/results/en&cmn/corpus-result-table. It consists of about 48,303 en tokens and 332,218 cmn_tokens


## Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository

## Acknowledgements
Thank the contributors and researchers whose work and ideas have inspired and influenced this project. Special thanks to the developers of the Transformer architecture and the Jieba and BPE tokenization techniques








