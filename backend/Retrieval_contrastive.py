import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
import json
import time
import timeit
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {"acc": acc, "prec0": prec0, "prec1": prec1, "rec0": rec0, "rec1": rec1, "auroc": auroc, "auprc": auprc,
            "minpse": minpse}

# Define model for training
class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        # The tokekenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to the
        # sequence, as well as compute the attention masks.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.BERTModel = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.linear = nn.Linear(768,2) # 768 is the dimension of the pooler_output, i.e., the CLS
        self.tau = 1

    def forward(self, Q, abstract):
        # here the input Q will contain the same query in each row (there are "batch size" of queries in total)
        # thus we only pick the first row to calculate the similarities
        token_Q = self.tokenizer(Q[0], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        CLS_Q= self.BERTModel(**token_Q).pooler_output

        #embedd the abstracts
        token_abstract = self.tokenizer(abstract, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        CLS_abstract = self.BERTModel(**token_abstract).pooler_output

        #calculate the similarity
        sim = torch.matmul(CLS_Q, torch.transpose(CLS_abstract,0,1)) / (torch.norm(CLS_Q)*torch.norm(CLS_abstract, dim=1))
        sim = sim/self.tau
        return sim

# Define model for testing
class NeuralNetwork_test(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork_test, self).__init__()
        # The tokekenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to the
        # sequence, as well as compute the attention masks.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.BERTModel = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.linear = nn.Linear(768,2) # 768 is the dimension of the pooler_output, i.e., the CLS
        self.tau = 1

    def forward(self, Q, embedded_abstract):
        # The module received the abstracts already embedded, therefore we do not need to reembedd them

        # change the type and shape of embeddings to match BERT's output
        size = len(embedded_abstract)
        embedded_abstract = torch.tensor(embedded_abstract).to(self.device)
        embedded_abstract = torch.reshape(embedded_abstract, [size, 768])

        #embedd the query
        token_Q = self.tokenizer(Q[0], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        CLS_Q = self.BERTModel(**token_Q).pooler_output

        #claculate the similarity
        sim = torch.matmul(CLS_Q, torch.transpose(embedded_abstract, 0, 1)) / (torch.norm(CLS_Q) * torch.norm(embedded_abstract, dim=1))
        sim = sim / self.tau
        return sim

def my_loss(output):
    loss = -torch.log(torch.exp(output[0][0])/torch.sum(torch.exp(output[0][1:])))
    return loss

def test_with_label_and_embeddings(queries, abstracts, embeddings, label, model, device, batch_size):
    size = len(label)
    model.eval()
    test_loss, correct = 0, 0
    batch_num = int(size / batch_size)  # discard the decimals
    pred_label = []
    threshhold = 0.9
    with torch.no_grad():
        for batch_index in range(batch_num):  # loop over the dataset batch by batch
            start = batch_index*batch_size
            Q = queries[start:(start + batch_size)]
            abstract = abstracts[start:(start + batch_size)]
            embedded_abstract = embeddings[start:(start + batch_size)]
            y = torch.tensor(label[start:(start + batch_size)]).to(device)

            sim = model(Q, embedded_abstract)
            boo_result = (sim > threshhold)
            scalar_result = (boo_result[0] * 1)
            correct += (y == scalar_result).sum()
            test_loss += my_loss(sim)
            pred_label.append(scalar_result.cpu().numpy())

            if batch_index % 100 == 0:
                current = batch_index * len(Q)
                print(f"loss: {test_loss/current:>7f} accuracy: {correct/current:>7f} [{current:>5d}/{size:>5d}]")

        if (batch_index + 1) * batch_size < size:  # still some left
            start = (batch_index + 1) * batch_size
            Q = queries[start:]
            abstract = abstracts[start:]
            y = torch.tensor(label[start:]).to(device)

            sim = model(Q, abstract)
            boo_result = (sim > threshhold)
            scalar_result = (boo_result[0] * 1)
            correct += (y == scalar_result).sum()
            test_loss += my_loss(sim)
            pred_label.append(scalar_result.cpu().numpy())

            if batch_index % 100 == 0:
                current = batch_index * len(Q)
                print(f"loss: {test_loss / current:>7f} accuracy: {correct / current:>7f} [{current:>5d}/{size:>5d}]")

    test_loss = test_loss / size
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pred_label = np.concatenate(np.array(pred_label))
    selected_abstracts = [abstracts[idx][1] for idx, _ in enumerate(pred_label) if pred_label[idx]==1]
    print_metrics_binary(label, pred_label, verbose=1)
    return pred_label, selected_abstracts

def find_embeddings(PMID): #find the embedded abstract using the PMID of the original abstract
    with open("embeddings_PMID_abstracts2.0.json", 'r') as json_file:  # training data
        PMID_abstracts_embeddings = json.load(json_file)
    size = len(PMID)
    print(size)
    embeddings = []
    for i in range(size):
        for j in range(size):
            if PMID[i] == PMID_abstracts_embeddings[1][j]:
                embeddings.append((PMID_abstracts_embeddings[0][j]))
                break
    #print("found",len(embeddings),"embeddings")
    return embeddings

def test_model_with_embeddings():
    # time_start = time.time()
    batch_size = 10
    # Get cpu or gpu device for training.
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    with open("Retreival_testingset4.0.json", 'r') as json_file:  # training data
        data_dict_test = json.load(json_file)

    # assign data, labels and PMID
    test_data = data_dict_test[0]
    test_label = data_dict_test[1]
    test_PMID = data_dict_test[2]
    test_embeddings = find_embeddings(test_PMID)
    size = len(test_data)
    queries = [0 for element in range(size)]
    abstracts = [0 for element in range(size)]
    for i in range(size):
        queries[i] = test_data[i][0]
        abstracts[i] = test_data[i][1]

    # Initialize the model
    # model = NeuralNetwork(device).to(device)
    model = NeuralNetwork_test(device).to(device)

    # load model
    model.load_state_dict(torch.load("Retrieval_model_contrastive2.0.pth", map_location=torch.device(device)),strict = False)
    #model.load_state_dict(torch.load("Retrieval_model_contrastive2.0.pth"))

    # test model on testing data
    pred_label, selected_abstracts = test_with_label_and_embeddings(queries, abstracts, test_embeddings, test_label, model, device, batch_size)

    return selected_abstracts

def train(quries, abstracts, label, model, optimizer, device, batch_size, num_epochs):
    size = len(label)
    batch_num = int(size / batch_size)  # discard the decimals
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for batch_index in range(batch_num): # loop over the dataset batch by batch
            start = batch_index*batch_size
            Q = quries[start:(start+batch_size)]
            abstract = abstracts[start:(start + batch_size)]
            #y = torch.tensor(label[start:(start+batch_size)]).to(device)
            # Compute prediction error
            sim = model(Q,abstract)
            loss = my_loss(sim)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                loss, current = loss, batch_index * len(Q)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return model

def train_model():
    time_start = time.time()
    batch_size = 10  # 16
    num_epochs = 5
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    with open("Retreival_trainingset4.0.json", 'r') as json_file:  # training data
        data_dict_train = json.load(json_file)

    #assign data, labels and PMID
    train_data = data_dict_train[0]
    train_label = data_dict_train[1]
    train_PMID = data_dict_train[2]
    size = len(train_data)
    queries = [0 for element in range(size)]
    abstracts = [0 for element in range(size)]
    for i in range(size):
        queries[i] = train_data[i][0]
        abstracts[i] = train_data[i][1]

    # Initialize the model
    model = NeuralNetwork(device).to(device)
    # define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # train the model
    model = train(queries, abstracts, train_label, model, optimizer, device, batch_size, num_epochs)
    # save model
    torch.save(model.state_dict(), "Retrieval_model_contrastive2.0.pth")
    print("Saved PyTorch Model State to model_train.pth")
    return model

def test(test_dataset, model, batch_size=4):
    dataset = test_dataset[0]
    pmids = test_dataset[1]
    size = len(dataset)
    for i in range(size):
        queries = dataset[i][0]
        abstracts = dataset[i][1]
        embeddings = dataset[i][2]
    model.eval()
    test_loss, correct = 0, 0
    batch_num = int(size / batch_size)  # discard the decimals
    pred_label = []
    threshhold = 0.9
    with torch.no_grad():
        for batch_index in range(batch_num):  # loop over the dataset batch by batch
            start = batch_index * batch_size
            Q = queries[start:(start + batch_size)]
            abstract = abstracts[start:(start + batch_size)]
            embedded_abstract = embeddings[start:(start + batch_size)]

            sim = model(Q, embedded_abstract)
            boo_result = (sim > threshhold)
            scalar_result = (boo_result[0] * 1)
            pred_label.append(scalar_result.cpu().numpy())

        if batch_index * batch_size < size:  # still some left
            start = (batch_index + 1) * batch_size
            Q = queries[start:]
            abstract = abstracts[start:]
            embedded_abstract = embeddings[start:]

            sim = model(Q, embedded_abstract)
            boo_result = (sim > threshhold)
            scalar_result = (boo_result[0] * 1)
            pred_label.append(scalar_result.cpu().numpy())

    pred_label = np.concatenate(np.array(pred_label))
    selected_abstracts = [dataset[idx][1] for idx, _ in enumerate(pred_label) if pred_label[idx]==1]
    selected_pmids = [pmids[idx] for idx,_ in enumerate(pred_label) if pred_label[idx] == 1]
    return pred_label, selected_abstracts, selected_pmids

def retrieval(test_data):
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Initialize the model
    model = NeuralNetwork_test(device).to(device)

    # load model
    #model.load_state_dict(torch.load("Retrieval_model_contrastive2.0.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load("Retrieval_model_contrastive2.0.pth"))

    # test model on testing data
    pred_label, selected_abstracts,selected_pmids = test(test_data, model)

    return selected_abstracts,selected_pmids

def retrieval_with_one_query(Q):
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Initialize the model
    model = NeuralNetwork_test(device).to(device)

    # load model
    model.load_state_dict(torch.load("./backend/Retrieval_model_contrastive2.0.pth", map_location=torch.device(device)), strict=False)

    # load embeddings
    with open("./backend/embeddings_PMID_abstracts2.0.json", 'r') as json_file:  # training data
        PMID_abstracts_embeddings = json.load(json_file)
    embeddings = PMID_abstracts_embeddings[0] # 0 for embeddings, 1 for PMID, 2 for abstracts
    pmids = PMID_abstracts_embeddings[1]
    abstracts = PMID_abstracts_embeddings[2]

    threshold = 0.9
    sim = model(Q, embeddings)
    boo_result = (sim > threshold)
    pred_label = (boo_result[0] * 1).cpu().numpy()
    selected_abstracts = [abstracts[idx] for idx, _ in enumerate(pred_label) if pred_label[idx] == 1]
    selected_pmids = [pmids[idx] for idx, _ in enumerate(pred_label) if pred_label[idx] == 1]

    return selected_abstracts,selected_pmids


def saving_embedding(abstract, model, device):
    token_abstract = model.tokenizer(abstract, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
        device)
    CLS_abstract = model.BERTModel(**token_abstract).pooler_output
    return CLS_abstract


def prepare_abstracts():
    with open('PIO_data_PMID_abstracts.json') as f:
        PMID_abstracts = json.load(f)
    size = len(PMID_abstracts)
    abstracts = [0 for element in range(size)]
    PMID = [0 for element in range(size)]
    for i in range(size):
        abstracts[i] = PMID_abstracts[i][1]
        PMID[i] = PMID_abstracts[i][0]

    embedded_abstracts = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    with torch.no_grad():
        # Initialize the model
        model = NeuralNetwork(device).to(device)
        model.load_state_dict(torch.load("Retrieval_model_contrastive2.0.pth", map_location=torch.device(device)))
        for j in range(size):
            embedded_abstract = saving_embedding(abstracts[j],model,device)
            print("embedded abstract",j,"is saved and its embedding",embedded_abstract)
            embedded_abstracts.append(embedded_abstract.cpu().numpy().tolist())
    return [embedded_abstracts, PMID, abstracts]

if __name__ == '__main__':
    # load data where test_data is the list of [query, abstract]
    # with open('Retreival_testingset4.0.json', 'r') as json_file:  # testing data
    #    data_dict_test = json.load(json_file)

    # call the retrieval function with input "test_data" and you can get the selected abstracts
    # retrieval(data_dict_test)
    selected_abstracts,selected_pmids = retrieval_with_one_query('cancer')
    print('done')
    #test_model_with_embeddings()


