import torch
import torch.nn as nn
from tqdm import tqdm
from contextlib import contextmanager
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import bcolz


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def train(epoch, loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    losses = AverageMeter()

    for batch_id, (X_num, X_cat, X_text, X_word, label) in pbar:
        optimizer.zero_grad()
        batch_size = X_num.size(0)

        # data = to_gpu(data)
        label = to_gpu(label)

        output = model(X_num, X_cat, X_text, X_word)
        loss = criterion(output, label)
        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)

        # Backward
        loss.backward()
        optimizer.step()

        pbar.set_description("[+] Epoch train {}, "
                             "Loss {:.4f} ({:.4f}), "
                             "RMSE {:.4f} ({:.4f})".format(
                              epoch, losses.val, losses.avg,
                              np.sqrt(losses.val), np.sqrt(losses.avg)
        ))
    print("[+] Epoch train {}, "
                             "Loss {:.4f} ({:.4f}), "
                             "RMSE {:.4f} ({:.4f})".format(
                              epoch, losses.val, losses.avg,
                              np.sqrt(losses.val), np.sqrt(losses.avg)
    ))


def test(epoch, test_loader, model, criterion):
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (X_num, X_cat, X_text, X_word, label) in pbar:
            batch_size = X_num.size(0)

            label = to_gpu(label)

            output = model(X_num, X_cat, X_text, X_word)
            loss = criterion(output, label)
            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            pbar.set_description("[+] Epoch test {}, "
                                 "Loss {:.4f} ({:.4f}), "
                                 "RMSE {:.4f} ({:.4f})".format(
                epoch, losses.val, losses.avg,
                np.sqrt(losses.val), np.sqrt(losses.avg)
            ))
        print("[+] Epoch test {}, "
              "Loss {:.4f} ({:.4f}), "
              "RMSE {:.4f} ({:.4f})\n".format(
            epoch, losses.val, losses.avg,
            np.sqrt(losses.val), np.sqrt(losses.avg)
        ))

        return losses.avg


def save_checkpoint(state, is_best, model_name, filename="model_ckp.pth.tar",
                    bestchkp="model_ckp_best.pth.tar"):
    import shutil
    """Saves checkpoint to disk"""
    directory = f"checkpoint/{model_name}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{directory}/{bestchkp}')


def load_checkpoint(checkpoint):
    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        ckp = torch.load(checkpoint)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint, ckp['epoch']))
        return ckp
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))
        return None


def create_empty_bcolz(n, name):
    return bcolz.carray(np.zeros((0, n), np.float32), chunklen=1, mode='w', rootdir=name)


def read_bcolz_data(name):
    return bcolz.open(name)


def save_bcolz(feature, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    bcolz.carray(feature, chunklen=1024, mode='w', rootdir=f"{root}/{name}")
    

def load_bcolz(root, name):
    if not os.path.exists(root):
        print(f"[+] Feature {name} does not exists")
        return None
    return bcolz.open(f"{root}/{name}")


def save_csv(df, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    df.to_csv(f"{root}/{name}", index=False)


def save_features(feature, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    np.save(f"{root}/{name}.npy", feature)


def load_features(root, name):
    if not os.path.exists(root):
        print(f"[+] Feature {name} does not exists")
        return None
    return np.load(f"{root}/{name}.npy")