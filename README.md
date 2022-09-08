# Tacotron 2 for Brazilian Portuguese Using GL as a Vocoder and CommonVoice Dataset

["Conversão Texto-Fala para o Português Brasileiro Utilizando Tacotron 2 com Vocoder Griffin-Lim"](https://biblioteca.sbrt.org.br/articles/2858) Paper published on SBrT 2021.

Repository containing pretrained Tacotron 2 models for brazilian portuguese using open-source implementations from [Rayhane-Mama](https://github.com/Rayhane-mamah/Tacotron-2) and [TensorflowTTS](https://github.com/Rayhane-mamah/Tacotron-2).

## Forked Tacotron 2 implementations

To train both models, modifications to adapt to brazilian portuguese were made at the original source code. They can be seen at their forked repositories:

* [Rayhane-Mama's Tacotron 2 fork](https://github.com/kobarion/Tacotron-2)
* [TensorflowTTS's Tacotron 2 fork](https://github.com/kobarion/TensorFlowTTS)

## Dataset used

Dataset used was originated from [Common Voice Corpus 4 portuguese dataset](https://commonvoice.mozilla.org/pt/datasets). Audio and text from the top speaker with around 6h of data was used and processed with the notebooks in this repository.

## Audio samples

For synthesized audio avaliation purposes, it was used 200 phonetically balanced sentences from SEARA, 1994. And their synthesized audios, mel spectrograms and alignment plots are available at [this google drive link](https://drive.google.com/drive/folders/1dNhpliv_3PYlNp2gRjCSizwoDQmjn9OP?usp=sharing)

## Trained models

The trained models can be found at [this google drive link](https://drive.google.com/drive/folders/10t9j4vOLtQbZkGtgbenMExI7luFkA96T?usp=sharing)

## Avaliation

The spreadsheet containing the avaliation of the 200 audio samples can be found [here](https://docs.google.com/spreadsheets/d/1NI0xKORUoe-Q-4MpUAdBdbhXGon1JiI91Ea0K2ak2f8/edit?usp=sharing)

## Steps to synthesize sentences using Rayhane-Mama's Tacotron 2 implementation

First, clone forked repository

`git clone https://github.com/kobarion/Tacotron-2`

Create conda environment

`conda create -y --name tacotron-2 python=3.6.9`

Install needed dependencies

`conda install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools`

Install libraries

`conda install --force-reinstall -y -q --name tacotron-2 -c conda-forge --file requirements.txt`

Enter conda environment

`conda activate tacotron-2`

Changes made to the repository in the forked version are in the following files. Check those to adapt to your dataset.
```
Tacotron-2/datasets/preprocessor.py
Tacotron-2/tacotron/synthesize.py
Tacotron-2/tacotron/utils/symbols.py
Tacotron-2/wavenet_vocoder/models/modules.py
Tacotron-2/hparams.py
Tacotron-2/preprocess.py
Tacotron-2/requirements.txt
```

Create a new folder with the trained model

`Tacotron2/logs-Tacotron/taco_best/{checkpoint/tacotron_model.ckpt-70000.index/data/meta}`

Create a text file with sentences to be synthesized

`sentences.txt`

Synthesize new sentences using the following command

`python synthesize.py --model=”Tacotron” --checkpoint=”best” --text_list=”sentences.txt” `


## To train TensorflowTTS's Tacotron 2 using the fork

With this notebook, resample CommonVoice samples to 22.05 kHz

`resample_wavs.ipynb`

The following files were modified to use brazilian portuguese

```
TensorFlowTTS/tensorflow_tts/processor/commonvoicebr.py
TensorFlowTTS/tensorflow_tts/configs/tacotron2.py
TensorFlowTTS/ttsexamples/tacotron2/conf/tacotron2.v1.yaml
TensorFlowTTS/tensorflow_tts/inference/auto_processor.py
TensorFlowTTS/preprocess/commonvoicebr_preprocess.yaml
TensorFlowTTS/notebooks/tacotron_synthesis.ipynb
```

Command to preprocess the dataset

```
tensorflow-tts-preprocess 
--rootdir ./commonvoicebr 
--outdir ./dump_commonvoicebr 
--config preprocess/commonvoicebr_preprocess.yaml 
--dataset commonvoicebr
``` 

Command to normalize the dataset

``` 
tensorflow-tts-normalize -
--rootdir ./commonvoicebr 
--outdir ./dump_commonvoicebr 
--config preprocess/commonvoicebr_preprocess.yaml 
--dataset commonvoicebr
``` 

Command to train TensorflowTTS's tacotron 2 model

``` 
CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/train_tacotron2.py
 --train-dir ./dump_commonvoicebr_16/train/ 
--dev-dir ./dump_commonvoicebr_16/valid/ 
--outdir ./ttsexamples/tacotron2/exp/train.tacotron2_finetune_32_r_2_wo_mx.v1/ 
--config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml 
--use-norm 1 
--pretrained ./ttsexamples/tacotron2/exp/train.tacotron2_pretrained.v1/model-65000.h5 --resume ""
``` 

Command to decode 

``` 
CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/decode_tacotron2.py 
--rootdir ./dump_commonvoicebr_16/valid/
 --outdir ./prediction/tacotron2_commonvoicebr-70k/ 
--checkpoint ./ttsexamples/tacotron2/exp/train.tacotron2_16_mx.v1/checkpoints/model-70000.h5 
--config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml --batch-size 16
``` 

For inference, use the following jupyter notebook

`tacotron_synthesis.ipynb`

## References

SEARA, I.. Estudo Estatístico dos Fonemas do Português Brasileiro Falado na Capital de Santa Catarina para elaboração de Frases Foneticamente Balanceadas. Dissertação de Mestrado, Universidade Federal de Santa Catarina, 1994.

SHEN, J. et al. Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions. 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, 2018, pp. 4779-4783.

## BibTeX

```
@inproceedings{Rosa2021,
  doi = {10.14209/sbrt.2021.1570727280},
  url = {https://doi.org/10.14209/sbrt.2021.1570727280},
  year = {2021},
  publisher = {Sociedade Brasileira de Telecomunica{\c{c}}{\~{o}}es},
  author = {Rodrigo K Rosa and Danilo Silva},
  title = {Convers{\~{a}}o Texto-Fala para o Portugu{\^{e}}s Brasileiro Utilizando Tacotron 2 com Vocoder Griffin-Lim},
  booktitle = {Anais do {XXXIX} Simp{\'{o}}sio Brasileiro de Telecomunica{\c{c}}{\~{o}}es e Processamento de Sinais}
}
```
