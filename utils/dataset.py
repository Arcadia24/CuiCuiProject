from PIL import Image
from torch.utils.data import Dataset

import pandas as pd
import os
import numpy as np
import torch
import torchvision.transforms as Tv

class BirdDataset(Dataset):
  def __init__(self, csv : pd.DataFrame, spec_pth : str, class_mapping : dict) -> None:
    self.spec = csv
    self.spec_pth = spec_pth
    self.class_mapping = class_mapping
    self.transform = Tv.Compose([Tv.ToTensor(),
                                 lambda x : x/255.0,
                                 lambda x : torch.vstack((x,x,x))])

  def __len__(self) -> int:
    return len(self.spec)  
  
  def Normalize(self, spec):
    spec = np.ndarray(spec, dtype = "float32")
    spec = spec / 255.0
    spec -= spec.min()
    spec /= spec.max()

  def __getitem__(self, idx: int):
    label = self.class_mapping[self.spec["en"].iloc[idx]]
    path =  os.path.join(self.spec_pth , self.spec["en"].iloc[idx], self.spec["filename"].iloc[idx])
    spec = Image.open(path)
    return self.transform(spec), label