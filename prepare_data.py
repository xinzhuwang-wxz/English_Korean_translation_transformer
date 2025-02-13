from train import data_folder
from utils import *

# download_data(data_folder=data_folder")

# note: if you can't download the data you can just click the
# https://raw.githubusercontent.com/Huffon/pytorch-transformer-kor-eng/93eea77b9562941813856f117aad7b2e4d3c01e8/data/corpus.csv
# and download the data manually and put it in the data folder

prepare_data(data_folder=data_folder)

# test

import youtokentome as yttm

bpe = yttm.BPE(model=os.path.join(data_folder, "bpe.model"))

sample_en = "Hello, how are you?"
sample_kr = "안녕하세요, 어떻게 지내세요?"

print("Tokenized English:", bpe.encode([sample_en], output_type=yttm.OutputType.SUBWORD))
print("Tokenized Korean:", bpe.encode([sample_kr], output_type=yttm.OutputType.SUBWORD))
