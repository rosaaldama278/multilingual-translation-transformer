<!--
 * @Author: Ting Aldama tingaldama278@gmail.com
 * @Date: 2024-07-29 13:07:38
 * @LastEditors: Ting Aldama tingaldama278@gmail.com
 * @LastEditTime: 2024-07-29 13:13:03
 * @FilePath: /multi-lang-translation/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Multi Language Translation Application

This project provides an translation model which includes all indo-enropean languages plus Chinese using a Transformer architecture. The project is organized to be easily downloaded, set up, and used for both inference and further training.

## Features

* Jieba + BPE tokenization for Chinese language:
  * Experimented with multiple tokenization methods, For Chinese language,  the combination of pre-tokenization with Jieba and pre trained BPE yielded the best results
* Trained with well-structured and high-quality parallel corpora
* Train BPE tokenizers for both source and target languages

## Setup

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- virtualenv (optional but recommended)

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd multi-lang-translation

   ```
2. **Install dependencies**:
   ```bash
   pip install -r resources/requirements.txt

   ```


### Running the Application

1. **Start the Flask Application**

    ```sh
    python app.py
    ```

2. **Access the Web Interface**

    Open your web browser and navigate to `http://127.0.0.1:5002`.

### Usage

1. **Enter Translation Details**

    - **Source Language**: Enter the source language code (e.g., `en` for English).
    - **Target Language**: Enter the target language code (e.g., `zh` for Chinese).
    - **Text to Translate**: Enter the text you want to translate.

2. **Submit the Form**

    Click the "Translate" button to get the translation.

3. **View the Translation**

    The translated text will be displayed on the same page.



### Training details:

* The script will read from data/source-lang- target_lang/ encoded_data.json
* train_loader and validation_loader will be created using MyDataset and collate_fn from utils/dataset.py
* The model checkpoint will be saved in the checkpoints/source_lang-target_lang

## Model Architecture

The translation model utilizes a Transformer architecture, which consists of an encoder and a decoder based on this famous paper: Attention Is All You Need. Different model settings were experimented and made the model slightly less complex compared to the original paper. The encoder processes the input English sentence, and the decoder generates the corresponding Chinese translation.

## Dataset

The dataset was obtained from this open-source nlp data: https://opus.nlpl.eu/results/en&cmn/corpus-result-table

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository

## Acknowledgements

Thank the contributors and researchers whose work and ideas have inspired and influenced this project. Special thanks to the developers of the Transformer architecture and the Jieba and BPE tokenization techniques
