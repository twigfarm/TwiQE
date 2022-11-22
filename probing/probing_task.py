from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import trange, tqdm
import time
import random

from transformers import AutoModel, AutoTokenizer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, sentences, labels, isTrain=True):
        self.sentences = sentences
        self.labels = labels
        self.isTrain = isTrain

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.sentences[idx]
        y = np.array([-1])
        if self.isTrain:
            y = self.labels[idx]
        return X, y


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 50)
        self.output_fc = nn.Linear(50, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = torch.sigmoid(self.input_fc(x))
        y_pred = self.output_fc(h_1)

        return y_pred, h_1


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        y = y.type(torch.LongTensor)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    y_true_bac = []
    y_pred_bac = []

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            y_true_bac.extend(y.tolist())

            y = y.type(torch.LongTensor)

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            y_pred_bac.extend(np.transpose(y_pred.argmax(1, keepdim=True).cpu().numpy()).tolist())

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    y_pred_bac_new = []
    for k in y_pred_bac:
        y_pred_bac_new.extend(k)

    # return epoch_loss / len(iterator), epoch_acc / len(iterator)

    return epoch_loss / len(iterator), balanced_accuracy_score(y_true_bac, y_pred_bac_new)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def prepare_data(data_dir):
    data = pd.read_csv(data_dir, sep='\t')
    output_dim = len(data['label'].unique())
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(data['sentence'], data['label'], stratify=data['label'],
                                                        test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test, output_dim


def st_transformer_embedding(model, X):
    X = X.tolist()
    X_emb = []
    for i in trange(0, len(X), 100):
        sents = X[i:min(i + 100, len(X))]
        embeddings = model.encode(sents).tolist()
        # 확인 후 수정
        for j in range(len(embeddings)):
            X_emb.append(embeddings[j])

    return np.array(X_emb)


def automodel_embedding(model, tokenizer, X):
    X = X.tolist()
    X_emb = []
    for i in trange(0, len(X), 100):
        sents = X[i:min(i + 100, len(X))]
        inputs = tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
        embeddings, _ = model(**inputs, return_dict=False)
        embeddings = embeddings.detach().tolist()
        # 확인 후 수정
        for j in range(len(embeddings)):
            X_emb.append(embeddings[j][0])

    return np.array(X_emb)


def train_logistic(X_train, X_test, y_train, y_test):
    model = LogisticRegression(penalty='l2', max_iter=1000, random_state=42, verbose=True)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def train_mlp(X_train, X_test, y_train, y_test, input_dim, output_dim):
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataset = MyDataset(X_train, y_train, isTrain=True)
    test_dataset = MyDataset(X_test, y_test, isTrain=True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = MLP(input_dim, output_dim)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in trange(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_dataloader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def main(data_dir, model_name, is_sbert, classifier):
    '''
    data_dir (str): 프로빙 태스크 데이터셋의 위치 경로
    model_name (str): 모델 이름
    is_sbert (bool): sentence_transformers에서 모델을 받아오는 경우 True
    classifier (str): 로지스틱 회귀일 경우 logistic, 다층 퍼셉트론일 경우 mlp
    output_dim (int): 데이터의 label 수
    '''
    # SentenceTransformer or Huggingface
    if is_sbert:
        model = SentenceTransformer(model_name)
        input_dim = model[-1].word_embedding_dimension
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_dim = model.pooler.dense.out_features

    # 파라미터 고정
    for para in model.parameters():
        para.requires_grad = False

    X_train, X_test, y_train, y_test, output_dim = prepare_data(data_dir)

    if is_sbert:
        X_train = st_transformer_embedding(model, X_train)
        X_test = st_transformer_embedding(model, X_test)

    else:
        X_train = automodel_embedding(model, tokenizer, X_train)
        X_test = automodel_embedding(model, tokenizer, X_test)

    y_train = np.array(y_train.tolist())
    y_test = np.array(y_test.tolist())

    # 분류기가 Logistic Regression
    if classifier == "logistic":
        train_logistic(X_train, X_test, y_train, y_test)

    # 분류기가 MLP
    elif classifier == "mlp":
        train_mlp(X_train, X_test, y_train, y_test, input_dim, output_dim)

    else:
        raise Exception("classifier에 잘못된 값이 전달되었습니다.")
