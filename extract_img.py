import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.models import *
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import argparse
import json
import pandas as pd
from tqdm import tqdm
from utils import *


def load_img(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except:
        return None


transform = transforms.Compose([
                         transforms.Resize((299, 299)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                     ])


class ImgDataset(data.Dataset):
    def __init__(self, root, df):
        super(ImgDataset, self).__init__()
        self.df = df
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fname = self.df.iloc[index]["image"]
        fname = f"{self.root}/{fname}.jpg"
        img = load_img(fname)
        if img is None:
            img = torch.zeros(3, 299, 299)
        else:
            img = transform(img)

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    args = parser.parse_args()

    config = json.load(open("config.json"))

    if args.mode == "train":
        df = pd.read_csv(config["train_csv"], usecols=["image"])
        root = '/media/deeplearning/E8DC7CDFDC7CAA08/Bac/avito/data/competition_files/train_jpg/'
    else:
        df = pd.read_csv(config["test_csv"], usecols=["image"])
        root = '/media/deeplearning/E8DC7CDFDC7CAA08/Bac/avito/test_jpg/data/competition_files/test_jpg/'

    # Define the dataset and loader
    dataset = ImgDataset(root, df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    bcolz_data = create_empty_bcolz(1000, f"extracted_features/{args.mode}_img")

    # Define the model
    model = inception_v3(pretrained=True)
    if torch.cuda.is_available():
        model.cuda()

    arr = []

    model.eval()
    with torch.no_grad():
        for batch_id, img in pbar:
            batch_size = img.size(0)
            img = to_gpu(img)
            output = model(img)
            arr.append(output[0].cpu().numpy())
    bcolz_data.append(np.asarray(arr))
    bcolz_data.flush()
