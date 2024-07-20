import torch
import math
from tqdm import tqdm
import os
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from Tools.tools_train import same_seeds
import time
import json
import csv
from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset


myseed = 87  # set a random seed for reproducibility
train_ratio = 0.9   # the ratio of data used for training, the rest will be used for validation
segment_len = int(128 * 2.0)
num_layers = 4
nhead = 2
data_dir = "./Dataset/Dataset"
result_path = './Result'
model_path = os.path.join(result_path, 'model.ckpt')
batch_size = 32
n_epochs = 60 * 6
early_stopping = n_epochs // 8
warm_epochs = int(n_epochs*0.05)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    same_seeds(myseed)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    loss_csv = open(os.path.join(result_path, 'loss.csv'), 'a+')
    loss_csv.write("Seed:{}, train_ratio: {}, segment_len:{}\n".format(myseed, train_ratio, segment_len))
    loss_csv.write("batch_size:{}, n_epochs: {}, warm_epochs:{}\n".format(batch_size, n_epochs, warm_epochs))
    loss_csv.flush()
    train_data = VoxDatasetTrain(data_dir=data_dir, train_ratio=train_ratio, seed=myseed, segment_len=segment_len)
    valid_data = VoxDatasetValid(data_dir=data_dir, train_ratio=train_ratio, seed=myseed, segment_len=segment_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, drop_last=False, num_workers=12, pin_memory=True)
    print("TrainSamples:{}, ValidSamples:{}\n".format(len(train_data), len(valid_data)))
    loss_csv.write("TrainSamples:{}, ValidSamples:{}\n".format(len(train_data), len(valid_data)))
    # model
    model = Classifier(n_spks=600, num_layers=num_layers, nhead=nhead).to(device)
    print('Parameters number is {}'.format(sum(param.numel() for param in model.parameters())))
    loss_csv.write('Parameters number is {}\n'.format(sum(param.numel() for param in model.parameters())))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    warmup_steps = len(train_loader) * warm_epochs
    total_steps = len(train_loader) * n_epochs
    print("warmup_steps:{}, total_steps:{}\n".format(warmup_steps, total_steps))
    loss_csv.write("warmup_steps:{}, total_steps:{}\n".format(warmup_steps, total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_accuracy = -1.0
    early_stop_count = 0
    iter = 0
    for epoch in range(n_epochs):
        train_item, train_acc_sum, train_loss_sum = 0, 0.0, 0.0
        # training
        model.train()
        cur_lr = optimizer.param_groups[0]['lr']
        st_time = time.time()
        for mels, labels in train_loader:
            mels = mels.to(device)
            labels = labels.to(device)
            outs = model(mels)
            # print(mels.shape, labels.shape, outs.shape)
            loss = criterion(outs, labels)
            # Get the speaker id with highest probability.
            preds = outs.argmax(1)
            # Compute accuracy.
            accuracy = (preds.detach() == labels.detach()).sum().item()
            train_loss_sum += loss.item() * labels.size(0)
            train_acc_sum += accuracy
            train_item += labels.size(0)
            # Updata model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            print('Epoch{:02d},Iter{:05d},Loss{:.6f},Lr{:.9f}'.format(epoch + 1, iter + 1, loss.item(), cur_lr))
            iter += 1
        # validation
        model.eval()  # set the model to evaluation mode
        valid_item, valid_acc_sum, valid_loss_sum = 0, 0.0, 0.0
        for mels, labels in valid_loader:
            mels = mels.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outs = model(mels)
                loss = criterion(outs, labels)
                # Get the speaker id with highest probability.
                preds = outs.argmax(1)
                # Compute accuracy.
                accuracy = (preds.cpu() == labels.cpu()).sum().item()
                valid_loss_sum += loss.item() * labels.size(0)
                valid_acc_sum += accuracy
                valid_item += labels.size(0)
        epo_time = time.time() - st_time
        print(f'[{epoch + 1:03d}/{n_epochs:03d}] [EpoTime: {epo_time:.0f}s] Train Acc: {train_acc_sum / train_item:3.5f} Loss: {train_loss_sum / train_item:3.5f} | '
              f'Val Acc: {valid_acc_sum / valid_item:3.5f} loss: {valid_loss_sum / valid_item:3.5f} Lr:{cur_lr:.9f}')
        loss_csv.write(f'[{epoch + 1:03d}/{n_epochs:03d}] [EpoTime: {epo_time:.0f}s] Train Acc: {train_acc_sum / train_item:3.5f} Loss: {train_loss_sum / train_item:3.5f} | '
              f'Val Acc: {valid_acc_sum / valid_item:3.5f} loss: {valid_loss_sum / valid_item:3.5f} Lr:{cur_lr:.9f}\n')
        loss_csv.flush()

        # if the model improves, save a checkpoint at this epoch
        if valid_acc_sum > best_accuracy:
            best_accuracy = valid_acc_sum
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {best_accuracy / valid_item:.5f}')
            loss_csv.write(f'saving model with acc {best_accuracy / valid_item:.5f}\n')
            loss_csv.flush()
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                loss_csv.write(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break

    # predict testing result
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8, collate_fn=inference_collate_batch)
    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(os.path.join(result_path, 'output.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.
        Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)
    outs = model(mels)
    loss = criterion(outs, labels)
    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())
    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        pbar.update(dataloader.batch_size)
        pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}", accuracy=f"{running_accuracy / (i+1):.2f}")
    pbar.close()
    model.train()
    return running_accuracy / len(dataloader)


class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1, num_layers=2, nhead=2):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=nhead, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        out = self.prenet(mels)  # batch size, length, d_model
        out = out.permute(1, 0, 2)  # out: length, batch size, d_model
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        # out = self.encoder_layer(out)
        out = self.encoder(out)  # batch size, length, d_model
        out = out.transpose(0, 1)  # batch size, length, d_model
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VoxDatasetTrain(Dataset):
    def __init__(self, data_dir='', train_ratio=0.9, seed=1213, segment_len=128):
        self.data_dir = data_dir
        random.seed(seed)

        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())  # 2 dicts: speaker2id & id2speaker
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                data.append([utterances["feature_path"], self.speaker2id[speaker]])

        # split training and validation data
        random.shuffle(data)
        train_len = int(len(data) * train_ratio)
        train_data = data[:train_len]
        self.train_data = []
        for i in range(len(train_data)):
            feat_path, speaker = train_data[i]
            # Load preprocessed mel-spectrogram.
            mel = torch.load(os.path.join(self.data_dir, feat_path))
            # Segmemt mel-spectrogram into "segment_len" frames.
            if len(mel) > segment_len:
                # Randomly get the starting point of the segment.
                start = random.randint(0, len(mel) - segment_len)
                # Get a segment with "segment_len" frames.
                mel = torch.FloatTensor(mel[start:start + segment_len])
                self.train_data.append({'mel': mel, 'speaker': torch.tensor(speaker).long()})
        # print('-'*100)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index]['mel'], self.train_data[index]['speaker']


class VoxDatasetValid(Dataset):
    def __init__(self, data_dir='', train_ratio=0.9, seed=1213, segment_len=128):
        self.data_dir = data_dir
        random.seed(seed)

        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())  # 2 dicts: speaker2id & id2speaker
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                data.append([utterances["feature_path"], self.speaker2id[speaker]])

        # split training and validation data
        random.shuffle(data)
        train_len = int(len(data) * train_ratio)
        valid_data = data[train_len:]
        self.valid_data = []
        for i in range(len(valid_data)):
            feat_path, speaker = valid_data[i]
            # Load preprocessed mel-spectrogram.
            mel = torch.load(os.path.join(self.data_dir, feat_path))
            mel = torch.FloatTensor(mel)
            self.valid_data.append({'mel': mel, 'speaker': torch.tensor(speaker).long()})
        # print('-'*100)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, index):
        return self.valid_data[index]['mel'], self.valid_data[index]['speaker']


# def collate_batch(batch):
#     # Process features within a batch.
#     """Collate a batch of data."""
#     mel, speaker = zip(*batch)
#     # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
#     mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
#     # mel: (batch size, length, 40)
#     return mel, torch.FloatTensor(speaker).long()


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)


if __name__ == '__main__':
    main()
    print(torch.__version__)
