# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
import time
from torchvision.models import resnet18

myseed = 612  # set a random seed for reproducibility
batch_size = 64
n_epochs = 50
learning_rate = 1e-5  # learning rate
weight_decay = 1e-5
result_path = './Result'
model_path = os.path.join(result_path, 'model.ckpt')  # the path where the checkpoint will be saved
early_stopping = n_epochs // 8

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),

    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.Resize(256),  # 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])


def main():
    same_seeds(myseed)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    loss_csv = open(os.path.join(result_path, 'loss.csv'), 'a+')
    loss_csv.write(
        "Seed:{}, N_epochs:{}, Batch:{}, Init_lr:{}, Wd:{}\n".format(myseed, n_epochs, batch_size, learning_rate,
                                                                     weight_decay))

    # The data set path needs to be added by yourself
    train_set = FoodDataset("./Dataset/food-11/train", tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    valid_set = FoodDataset("./Dataset/food-11/valid", tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_set = FoodDataset("./Dataset/food-11/test", tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    loss_csv.write("TrainSamples:{}, ValidSamples:{}, MaxIter:{}\n".format(len(train_set), len(valid_set),
                                                                           len(train_loader) * n_epochs))

    # Initialize a model, and put it on the device specified.
    # model = Classifier().to(device)
    model = resnet18(pretrained=False, progress=False, num_classes=11).to(device)
    print('Parameters number is {}'.format(sum(param.numel() for param in model.parameters())))
    loss_csv.write('Parameters number is {}\n'.format(sum(param.numel() for param in model.parameters())))
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * n_epochs, eta_min=1e-6)
    trainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, loss_csv)
    predict(test_set, test_loader, model)


def trainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, loss_csv):
    # Initialize trackers, these are not parameters and should not be changed
    early_stop_count = 0
    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        cur_lr = optimizer.param_groups[0]['lr']
        st_time = time.time()
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters with computed gradients.
            optimizer.step()
            scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        # Validation
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))
                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        epo_time = time.time() - st_time
        # Print the information.
        print(
            f"[{epoch + 1:03d}/{n_epochs:03d}] [EpoTime: {epo_time:.0f}s] [Train | loss = {train_loss:.5f}, acc = {train_acc:.5f}] "
            f"[Valid | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}] Lr:{cur_lr:.9f}")
        loss_csv.write(
            f"[{epoch + 1:03d}/{n_epochs:03d}] [EpoTime: {epo_time:.0f}s] [Train | loss = {train_loss:.5f}, acc = {train_acc:.5f}] "
            f"[Valid | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}] Lr:{cur_lr:.9f}\n")
        loss_csv.flush()
        # save models
        if valid_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {valid_acc:.5f}')
            loss_csv.write(f'saving model with acc {valid_acc:.5f}\n')
            loss_csv.flush()
            best_acc = valid_acc
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                loss_csv.write(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label

        return im, label


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict(test_set, test_loader, model):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    # create test csv
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = prediction
    df.to_csv(os.path.join(result_path, 'submission.csv'), index=False)


if __name__ == '__main__':
    main()
    print(torch.__version__)
