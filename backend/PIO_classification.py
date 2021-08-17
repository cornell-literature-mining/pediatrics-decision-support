# ==============================================
#   Quantify the Classification Performance
# ==============================================
import torch
import numpy as np
from transformers import BertForTokenClassification, BertTokenizer
import pickle
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


# ====================================
#           load model
# ====================================
tag_values = ['0', 'Pop', 'Int', 'Out', 'PAD']
tag2idx = {'0': 0, 'Pop': 1, 'Int': 2, 'Out': 3, 'PAD': 4}
MXLEN = 512
print(tag2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

model = torch.load("model_PIO_dropout_5epoch_entire.pth")
model.eval()  # this is important
model = model.to(device)  # this is important

tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=False)

# ====================================
#            load data
# ====================================
# three lists annotated by three professionals in this gold
with open('PIO_data_test.json', 'r') as json_file:  # gold data to test
    gold_abstract_list = json.load(json_file)
data = gold_abstract_list[0]
abstracts = data['doc']
labels = data['label']

# preprocess the data to preserve the labels when tokenizing
tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(tokenizer, sent, labs)
    for sent, labs in zip(abstracts, labels)
]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
tokenized_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
input_ids = [tokenizer.convert_tokens_to_ids(tokenized_txt) for tokenized_txt in tokenized_texts]
input_ids = pad_sequences(input_ids,
                          maxlen=MXLEN, dtype="long", value=0.0,
                              truncating="post", padding="post")
true_labels = pad_sequences(tokenized_labels,
                     maxlen=MXLEN, dtype=object, value="PAD",
                            padding="post", truncating="post")

# attention mask will improve the classification performance
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

# ====================================
#             testing
# ====================================
accuracy_values, F1_values, precision_values, recall_values = [], [], [], []
predictions = []
i = 0
for input_id, attention_mask in zip(input_ids,attention_masks):
# for input_id in input_ids:
    i = i + 1
    if i % 20 == 0:
        print("{}".format(i))

    # tokenized_sentence = tokenizer.encode(abstract, padding=True, truncation=True, max_length=512)

    # input_ids = tokenizer.convert_tokens_to_ids(abstract)
    input_id = torch.tensor([input_id.tolist()]).cuda()
    attention_mask = torch.tensor([attention_mask]).cuda()
    with torch.no_grad():
        output = model(input_id,attention_mask=attention_mask)
        #output = model(input_id)
    # print(len(output[0].to('cpu').numpy()))
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    predictions.append(label_indices)

# predict_tags = [tag_values[l_i] for p in predictions
#                   for l_i in p[0] if tag_values[l_i] != "PAD"]
predict_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p[0], l) if l_i != "PAD"]
true_tags = [l_i for labels in true_labels
                    for l_i in labels if l_i != "PAD"]

accuracy = accuracy_score(true_tags,predict_tags)
print("Validation Accuracy: {}".format(accuracy))

precision = precision_score(true_tags,predict_tags,labels=['Pop','Int','Out'],average='micro')
print("Validation Precision-Score: {}".format(precision))

recall = recall_score(true_tags,predict_tags,labels=['Pop','Int','Out'],average='micro')
print("Validation Recall-Score: {}".format(recall))

F1 = f1_score(true_tags,predict_tags,labels=['Pop','Int','Out'],average='micro')
print("Validation F1-Score: {}".format(F1))

print("Confusion Matirx:\n {}".format(confusion_matrix(true_tags,predict_tags,labels=['Pop','Int','Out','0'])))

