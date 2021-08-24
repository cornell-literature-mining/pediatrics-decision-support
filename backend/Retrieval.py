from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
import json
import time
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import numpy as np
from . import File_loading as fl


# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        # The tokekenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to the
        # sequence, as well as compute the attention masks.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.BERTModel = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.linear = nn.Linear(768,2) # 768 is the dimension of the pooler_output, i.e., the CLS

    def forward(self, x):
        token_x = self.tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        CLS_x = self.BERTModel(**token_x).pooler_output # pooler_output is the CLS, the representation of the whole sentence
        logits = self.linear(CLS_x)
        return logits


def train(dataset, label, model, loss_fn, optimizer, device, batch_size, num_epochs):
    size = len(label)
    batch_num = int(size / batch_size)  # discard the decimals
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for batch_index in range(batch_num): # loop over the dataset batch by batch
            start = batch_index*batch_size
            X = dataset[start:(start+batch_size)]
            y = torch.tensor(label[start:(start+batch_size)]).to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                loss, current = loss.item(), batch_index * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # save model
        torch.save(model.state_dict(), "model_train_{}.pth".format(epoch))
        print("Saved PyTorch Model State to model_train_{}.pth".format(epoch))

def test_with_label(dataset, label, loss_fn, model, device, batch_size=4):
    size = len(label)
    model.eval()
    test_loss, correct = 0, 0
    batch_num = int(size / batch_size)  # discard the decimals
    pred_label = []
    with torch.no_grad():
        for batch_index in range(batch_num):  # loop over the dataset batch by batch
            start = batch_index * batch_size
            X = dataset[start:(start + batch_size)]
            y = torch.tensor(label[start:(start + batch_size)]).to(device)

            pred = model(X)
            pred_label.append(pred.argmax(1).cpu().numpy())
            confidence = F.softmax(pred, dim=-1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch_index * batch_size < size:  # still some left
            start = (batch_index + 1) * batch_size
            X = dataset[start:]
            y = torch.tensor(label[start:]).to(device)

            pred = model(X)
            pred_label.append(pred.argmax(1).cpu().numpy())
            confidence = F.softmax(pred, dim=-1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pred_label = np.concatenate(np.array(pred_label))
    selected_abstracts = [dataset[idx][1] for idx, _ in enumerate(pred_label) if pred_label[idx]==0]
    return pred_label, selected_abstracts

def test(dataset, model, batch_size=4):
    size = len(dataset)
    model.eval()
    test_loss, correct = 0, 0
    batch_num = int(size / batch_size)  # discard the decimals
    pred_label = []
    with torch.no_grad():
        for batch_index in range(batch_num):  # loop over the dataset batch by batch
            start = batch_index * batch_size
            X = dataset[start:(start + batch_size)]

            pred = model(X)
            pred_label.append(pred.argmax(1).cpu().numpy())

        if batch_index * batch_size < size:  # still some left
            start = (batch_index + 1) * batch_size
            X = dataset[start:]

            pred = model(X)
            pred_label.append(pred.argmax(1).cpu().numpy())

    pred_label = np.concatenate(np.array(pred_label))
    selected_abstracts = [dataset[idx][1] for idx, _ in enumerate(pred_label) if pred_label[idx]==0]
    return pred_label, selected_abstracts


def retrieval(test_data):
    # time_start = time.time()
    # batch_size = 5  # 16
    # num_epochs = 5
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # load data, each dataset (data_balance_test.json, data_balance_vali.json, data_balance_train.json) is a dict
    # including train_data, train_label, test_data, test_label
    # with open('data_balance_train.json', 'r') as json_file:  # training data
    #     data_dict_train = json.load(json_file)

    # with open('data_balance_test.json', 'r') as json_file:  # testing data
    #     data_dict_test = json.load(json_file)

    # [query, abstract1] 1
    # [query, abstract2] 0
    # abstracts " bla bla bla "

    # train_data = data_dict_train['train_data']
    # train_label = data_dict_train['train_label']

    # test_data = data_dict_test['test_data']
    # test_label = data_dict_test['test_label']

    # Initialize the model
    model = NeuralNetwork(device).to(device)

    # define loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train the model
    # train(train_data, train_label, model, loss_fn, optimizer, device, batch_size, num_epochs)

    # save model
    # torch.save(model.state_dict(), "model_train.pth")
    # print("Saved PyTorch Model State to model_train.pth")

    # load model
    # load saved model parameters trained by the training data from the train reviews.
    # "model_test.pth" is trained from the training data from testing reviews
    # "model_train.pth" is trained from the training data from training reviews
    model.load_state_dict(fl.torch_load("model_train.pth"))

    # test model on testing data
    # pred_label, selected_abstracts = test_with_label(test_data, test_label, loss_fn, model, device, batch_size)
    pred_label, selected_abstracts = test(test_data, model)

    # time_end = time.time()
    # print('time cost', (time_end - time_start)/60, 'min')

    # print('done!')
    return selected_abstracts

if __name__ == '__main__':
    # load data where test_data is the list of [query, abstract]
    with open('data_balance_test.json', 'r') as json_file:  # testing data
        data_dict_test = json.load(json_file)
    test_data = data_dict_test['test_data']

    # call the retrieval function with input "test_data" and you can get the selected abstracts
    selected_abstracts = retrieval(test_data)
