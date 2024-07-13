from torch.utils.data import DataLoader
import gc
import torch
import os
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset
import random
from tqdm import tqdm


# data prarameters
concat_nframes = 3+8+8+8+8   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 307       # random seed
batch_size = 512           # batch size
num_epoch = 30        # the number of training epoch
learning_rate = 1e-5     # learning rate
result_path = './Result'
model_path = os.path.join(result_path, 'model.ckpt')  # the path where the checkpoint will be saved
early_stopping = num_epoch // 4

# model parameters
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 1 * 8         # the number of hidden layers
hidden_dim = 64 * 4          # the hidden dim
drop_rate = 0.5


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    same_seeds(seed)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    loss_csv = open(os.path.join(result_path, 'loss.csv'), 'a+')
    loss_csv.write("N_epochs:{}, Batch:{}, Init_lr:{}\n".format(num_epoch, batch_size, learning_rate))
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./Dataset/libriphone/feat', phone_path='./Dataset/libriphone',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', feat_dir='./Dataset/libriphone/feat', phone_path='./Dataset/libriphone',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)
    test_X = preprocess_data(split='test', feat_dir='./Dataset/libriphone/feat', phone_path='./Dataset/libriphone',
                             concat_nframes=concat_nframes)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)
    test_set = LibriDataset(test_X, None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    loss_csv.write("TrainSamples:{}, ValidSamples:{}, MaxIter:{}\n".format(train_X.shape, val_X.shape, len(train_loader)*num_epoch))
    trainer(train_loader, val_loader, train_set, val_set, loss_csv)
    predict(test_loader)


def trainer(train_loader, val_loader, train_set, val_set, loss_csv):
    # create model, define a loss function, and optimizer
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, drop_rate=drop_rate).to(device)
    print('Parameters number is {}'.format(sum(param.numel() for param in model.parameters())))
    loss_csv.write('Parameters number is {}\n'.format(sum(param.numel() for param in model.parameters())))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    early_stop_count = 0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        cur_lr = optimizer.param_groups[0]['lr']
        st_time = time.time()
        for i, batch in enumerate(train_loader):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cur_lr = optimizer.param_groups[0]['lr']

            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
            print('Epoch{:02d},Iter{:05d},Loss{:.6f},Lr{:.9f}'.format(epoch+1, i+1, loss.item(), cur_lr))

        # validation
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item()

        epo_time = time.time() - st_time
        print(f'[{epoch + 1:03d}/{num_epoch:03d}] [EpoTime: {epo_time:.0f}s] Train Acc: {train_acc / len(train_set):3.5f} Loss: {train_loss / len(train_loader):3.5f} | '
              f'Val Acc: {val_acc / len(val_set):3.5f} loss: {val_loss / len(val_loader):3.5f} Lr:{cur_lr:.9f}')
        loss_csv.write(f'[{epoch + 1:03d}/{num_epoch:03d}] [EpoTime: {epo_time:.0f}s] Train Acc: {train_acc / len(train_set):3.5f} Loss: {train_loss / len(train_loader):3.5f} | '
              f'Val Acc: {val_acc / len(val_set):3.5f} loss: {val_loss / len(val_loader):3.5f} Lr:{cur_lr:.9f}\n')
        loss_csv.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {best_acc / len(val_set):.5f}')
            loss_csv.write(f'saving model with acc {best_acc / len(val_set):.5f}\n')
            loss_csv.flush()
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                loss_csv.write(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, drop_rate=0):
        super(Classifier, self).__init__()
        self.lstm = nn.GRU(input_size=39, hidden_size=hidden_dim, num_layers=hidden_layers, bidirectional=True,
                            batch_first=True, dropout=drop_rate)
        self.fc = nn.Linear(hidden_dim*2, output_dim)


    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1, 39)  # b,l*hin ==>b,l,hin
        x, h_n = self.lstm(x, None)  # x:b,l,h  h_n:d*num_layer,b,h
        # x = x[:, -1, :]  # final state of final layer  ==>  x:b,h
        x_fd = h_n[-2, :, :]  # forward final state of final layer  ==>  x:b,h
        x_bd = h_n[-1, :, :]  # backward final state of final layer  ==>  x:b,h
        out = self.fc(torch.cat([x_fd, x_bd], dim=-1))
        return out


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)



def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, seed=1213):
    class_num = 41  # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        random.seed(seed)
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
        print(y.shape)
        return X, y
    else:
        return X


def predict(test_loader):
    # load model
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    pred = np.array([], dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            features = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
    with open(os.path.join(result_path, 'prediction.csv'), 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))


if __name__ == '__main__':
    main()
    print(torch.__version__)
