from audiomentations import Compose, AddGaussianNoise, LowPassFilter, TanhDistortion
from PIL import Image
from tqdm import tqdm

import argparse
import librosa
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as Tv




def get_spectrograms(filepath : str, 
                     primary_label : str, 
                     filename : str, output_dir : str, 
                     args : argparse.Namespace, 
                     augmentations : bool):
    """Get spectrograms from audio files

    Args:
        filepath (str): filepath to audio file
        primary_label (str): primary label for audio file
        filename (str): name of audio file
        output_dir (str): output directory for spectrograms
        args (argparse.Namespace): arguments
        augmentations (bool): whether to apply augmentations

    Returns:
        list: list of spectrogram names
        list: list of primary labels
    """

    # Open the file with librosa (limited to the first 15 seconds)
    sig, rate = librosa.load(os.path.join(filepath, primary_label,filename), sr=args.sample_rate, offset=None, duration = 30)
    
    if augmentations:
        augment = Tv.Compose([
            Tv.ToTensor(),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
            LowPassFilter(min_cutoff_freq=100,
                          max_cutoff_freq=7500,
                          p = 0.4),
            TanhDistortion(p=0.4),
            lambda x : x.numpy(),
        ])
        
    
    # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(sig), int(args.signal_length * args.sample_rate)):
        split = sig[i:i + int(args.signal_length * args.sample_rate)]

        # End of signal?
        if len(split) < int(args.signal_length * args.sample_rate):
            break
        
        sig_splits.append(split)
        
    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    saved_specname = []
    ens = []
    for chunk in sig_splits:
        
        mel_spec = librosa.feature.melspectrogram(y=chunk, 
                                                  sr=args.sample_rate, 
                                                  n_fft=1024, 
                                                  hop_length=args.hop_length, 
                                                  n_mels=args.num_mels, 
                                                  fmin=args.fmin, 
                                                  fmax=args.fmax)
    
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        # Save as image file
        save_dir = os.path.join(output_dir, primary_label)
        spec_name = filename[:-4] + '_' + str(s_cnt) + '.png'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, spec_name)
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)
        saved_specname.append(spec_name)
        ens.append(primary_label)
        s_cnt += 1
        
        
    return saved_specname, ens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_path',
                        type = str,
                        default = "dataset/train_metadata.csv",
                        help = "Path to csv file with labels, audio files name")
    parser.add_argument('--speccsv_path',
                        type=str,
                        default = "dataset/spec.csv",
                        help = "Path to csv file with spectrograms")
    parser.add_argument('--audio_dir',
                        type = str,
                        default = "dataset/train_short_audio/",
                        help = "Path to audio directory")
    parser.add_argument('--spectrogram_dir',
                        type = str,
                        default = "dataset/spectrograms/",
                        help = "Path to spectrogram directory")
    parser.add_argument('--augment',
                        type = bool,
                        default=False,
                        help = "Whether to apply augmentations")
    parser.add_argument('--rating_lim',
                        type=int,
                        default=4,
                        help="Minimum rating for audio files")
    parser.add_argument('--sample_rate',
                        type = int, 
                        default = 32000,
                        help ="Sample rate of audio files")
    parser.add_argument('--signal_length',
                        type = int,
                        default = 5,
                        help = "Length of audio files in seconds")
    parser.add_argument('--hop_length',
                        type = int,
                        default = 512,
                        help = "Hop length for mel spectrogram")
    parser.add_argument('--num_mels',
                        type = int,
                        default = 128,
                        help = "Number of mel bands")
    parser.add_argument('--fmin',
                       type = int,
                       default = 500,
                       help = "Minimum frequency for mel spectrogram")
    parser.add_argument('--fmax',
                        type = int,
                        default= 12500,
                        help = "Maximum frequency for mel spectrogram")
    args = parser.parse_args()
    
    # Read csv file
    audio = pd.read_csv(args.csv_path)
    
    #clear the csv file
    #filter (rating)
    temp_str = 'rating>='+str(args.rating_lim)
    audio = audio.query(temp_str)
    print('\nRATING LIMITER APPLIED:')
    print(f'[SUBSET]: {audio.values.shape} : LABELS {len(audio.primary_label.value_counts())}')

    
    spec_name = []
    ens = [] 
    for row in tqdm(audio.iterrows(), total = audio.shape[0]):
        en = row[1][0]
        filename = row[1][9]
        spec_name_, en_ = get_spectrograms(args.audio_dir, en, filename, args.spectrogram_dir, args, args.augment)
        spec_name+=spec_name_
        ens+=en_
    spec = pd.DataFrame({"filename" : spec_name, "en" : ens})
    spec.to_csv(os.path.join(args.speccsv_path), index = False)