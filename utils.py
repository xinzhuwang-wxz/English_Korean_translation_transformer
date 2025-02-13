import torch
import os
import requests
import wget
import tarfile
import shutil
import codecs
import youtokentome
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_data(data_folder):
    """
    Downloads the corpus.csv from https://github.com/Huffon/pytorch-transformer-kor-eng/tree/93eea77b9562941813856f117aad7b2e4d3c01e8/data

    :param data_folder: the folder where the files will be downloaded, the name data is recommended

    """
    raw_url = "https://raw.githubusercontent.com/Huffon/pytorch-transformer-kor-eng/93eea77b9562941813856f117aad7b2e4d3c01e8/data/corpus.csv"
    # make the data folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    # download the csv
    try:
        response = requests.get(raw_url)
        response.raise_for_status()

        # save the csv
        csv_file_path = os.path.join(data_folder, "corpus.csv")
        with open(csv_file_path, 'wb') as f:
            f.write(response.content)

        print(f"\nData successfully downloaded and saved to {csv_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")


def prepare_data(data_folder):
    """
    prepares the data, trains a Byte-Pair Encoding (BPE) model.

    :param data_folder: the folder where the files were downloaded, the name data is recommended

    """

    from sklearn.model_selection import train_test_split
    import pandas as pd
    import os
    import codecs

    df = pd.read_csv(os.path.join(data_folder, "corpus.csv"))

    # delete rows with \ in them
    df = df[~df['korean'].astype(str).str.contains(r'\\', regex=True)]
    df = df[~df['english'].astype(str).str.contains(r'\\', regex=True)]

    train_size = 0.2
    val_size = 0.3
    test_size = 0.5

    train_data, temp_data = train_test_split(df, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=42)


    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

        # split the data and save

    with codecs.open(os.path.join(data_folder, "train.kr"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_data['korean']))  # train_data['korean'] is a list of korean sentences, join them with "\n" separator,become a string like "sentence1\nsentence2\n..."
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_data['english']))

    with codecs.open(os.path.join(data_folder, "val.kr"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_data['korean']))
    with codecs.open(os.path.join(data_folder, "val.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_data['english']))

    with codecs.open(os.path.join(data_folder, "test.kr"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_data['korean']))
    with codecs.open(os.path.join(data_folder, "test.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_data['english']))

    with codecs.open(os.path.join(data_folder, "train.enkr"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_data['english']) + "\n" + "\n".join(train_data['korean']))

    # with codecs.open(os.path.join(data_folder, "train.enkr"), "w", encoding="utf-8") as f:
    #     f.write("\n".join([f"{e} ||| {k}" for e, k in zip(train_data['english'], train_data['korean'])]))

    print(f"Data successfully split and saved in {data_folder}")

    youtokentome.BPE.train(data=os.path.join(data_folder, "train.enkr"), vocab_size=10000,
                           model=os.path.join(data_folder, "bpe.model"))

    print("\nBPE is DONE!\n")


def get_positional_encoding(d_model, max_length=100):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    # lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

    return lr


def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer [model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix]
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)


def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



