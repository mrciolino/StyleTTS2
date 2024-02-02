from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from pathlib import Path
from munch import Munch
import torchaudio
import subprocess
import phonemizer
import requests
import torch
import tqdm
import sys

from .phoneme import PhonemeConverterFactory

SINGLE_INFERENCE_MAX_LEN = 420

# global for mel normalization
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def log_print(message, logger):
    logger.info(message)
    print(message)


def phoneme_check(defined_converter):
    try:
        if defined_converter == "gruut":
            phoneme_converter = PhonemeConverterFactory.load_phoneme_converter(defined_converter)
            phoneme_name = "gruut"
        else:
            subprocess.run(["espeak-ng", "--voices"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            phoneme_converter = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
            phoneme_name = "espeak"
    except FileNotFoundError:
        if "win" in sys.platform and defined_converter == "espeak":
            phoneme_name = "gruut"
            phoneme_converter = PhonemeConverterFactory.load_phoneme_converter(phoneme_name)
            print("Warning: espeak phoneme converter is not supported on Windows. Using gruut instead.")
        else:
            raise RuntimeError("espeak-ng is not installed on your system. Run `sudo apt install espeak-ng`.")
    return phoneme_name, phoneme_converter


def download_file(url, local_path, verbose=False):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(local_path).exists():
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(total=total_size, desc=f"Downloading {Path(local_path).name}", unit="B", unit_scale=True)
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(8192)
        pbar.close()
    else:
        if verbose:
            print(f"File already exists at {local_path}. Skipping download.")


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def segment_text(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=SINGLE_INFERENCE_MAX_LEN,
        chunk_overlap=0,
        length_function=len,
    )
    segments = splitter.split_text(text)
    return segments
