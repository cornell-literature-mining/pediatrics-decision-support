# ==============================================
#         extract the PICO element
# ==============================================
import torch
import numpy as np
from transformers import BertForTokenClassification, BertTokenizer
import pickle
import json
from itertools import groupby
import Levenshtein


def simple(corpus, distance):  # remove the too similar strings in a list
    words = []
    while corpus:
        center = corpus[0]
        words.append(center)
        corpus = [word for word in corpus
                  if Levenshtein.distance(center, word) >= distance]
    return words


def list2str(li):
    return ' '.join([str(elem) for elem in li])


def remove_duplicate(li):
    li = list(set(li))
    return li


def present_PIO(tokens, labels):
    result = [list(y) for x, y in groupby(labels)]
    # print(result)
    P_list, I_list, O_list = [], [], []
    index = 0
    for group in result:
        if group[0] == "Pop":
            P_list.append(tokens[index:index + len(group)])
        elif group[0] == "Int":
            I_list.append(tokens[index:index + len(group)])
        elif group[0] == "Out":
            O_list.append(tokens[index:index + len(group)])
        index = index + len(group)

    # make each elment (list) to string
    #     P_list = [list2str(li) for li in P_list]
    #     I_list = [list2str(li) for li in I_list]
    #     O_list = [list2str(li) for li in O_list]

    P_list = simple(remove_duplicate([list2str(li) for li in P_list]), distance=3)
    I_list = simple(remove_duplicate([list2str(li) for li in I_list]), distance=3)
    O_list = simple(remove_duplicate([list2str(li) for li in O_list]), distance=3)

    return P_list, I_list, O_list

def extract_PIO(abstract):
    # ====================================
    #           load model
    # ====================================

    tag_values = ['0', 'Pop', 'Int', 'Out', 'PAD']
    tag2idx = {'0': 0, 'Pop': 1, 'Int': 2, 'Out': 3, 'PAD': 4}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = torch.load("model_PIO_dropout_5epoch_entire.pth")
    model.eval()  # this is important
    model = model.to(device)  # this is important

    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=False)

    # test_sentence = """
    # Background: Current strategies for preventing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are limited to nonpharmacologic interventions. Hydroxychloroquine has been proposed as a postexposure therapy to prevent coronavirus disease 2019 (Covid-19), but definitive evidence is lacking.\n\nMethods: We conducted an open-label, cluster-randomized trial involving asymptomatic contacts of patients with polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia, Spain. We randomly assigned clusters of contacts to the hydroxychloroquine group (which received the drug at a dose of 800 mg once, followed by 400 mg daily for 6 days) or to the usual-care group (which received no specific therapy). The primary outcome was PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary outcome was SARS-CoV-2 infection, defined by symptoms compatible with Covid-19 or a positive PCR test regardless of symptoms. Adverse events were assessed for up to 28 days.\n\nResults: The analysis included 2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020. A total of 1116 contacts were randomly assigned to receive hydroxychloroquine and 1198 to receive usual care. Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]). In addition, hydroxychloroquine was not associated with a lower incidence of SARS-CoV-2 transmission than usual care (18.7% and 17.8%, respectively). The incidence of adverse events was higher in the hydroxychloroquine group than in the usual-care group (56.1% vs. 5.9%), but no treatment-related serious adverse events were reported.\n\nConclusions: Postexposure therapy with hydroxychloroquine did not prevent SARS-CoV-2 infection or symptomatic Covid-19 in healthy persons exposed to a PCR-positive case patient. (Funded by the crowdfunding campaign YoMeCorono and others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).
    # """

    tokenized_sentence = tokenizer.encode(abstract, padding=True, truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []

    skip_flag = 0  # a flag to show whether need to skip the next loop
    for idx, (token, label_idx) in enumerate(zip(tokens, label_indices[0])):
        if skip_flag == 1:
            skip_flag = 0
            continue
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]  # attach to the previous token
        elif token == "-" and (idx + 1) < len(tokens) and label_idx != 0:
            new_tokens[-1] = new_tokens[-1] + token + tokens[idx + 1]  # join the previous and latter tokens
            skip_flag = 1
        elif token == "(" and (idx + 1) < len(tokens) and label_idx != 0:
            new_tokens.append(token + tokens[idx + 1])  # attach to the latter token
            new_labels.append(tag_values[label_idx])
            skip_flag = 1
        elif token == ")" and label_idx != 0:  # label_idx != 0 means it is P, I or O
            new_tokens[-1] = new_tokens[-1] + token  # attach to the previous token
        elif token == "," and label_idx != 0:
            new_tokens[-1] = new_tokens[-1] + token  # attach to the previous token
        elif token == "." and label_idx != 0:
            new_tokens[-1] = new_tokens[-1] + token
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    P_list, I_list, O_list = present_PIO(new_tokens, new_labels)

    print("Population:{}\n".format(P_list))
    print("Intervention:{}\n".format(I_list))
    print("Outcome:{}\n".format(O_list))

    for token, label in zip(new_tokens, new_labels):
        if label == "0":
            print(token, end=" ")
        else:
            print("{}({})".format(token, label), end=" ")
    return P_list, I_list, O_list

if __name__ == '__main__':
    test_sentence = """
    Background: Current strategies for preventing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are limited to nonpharmacologic interventions. Hydroxychloroquine has been proposed as a postexposure therapy to prevent coronavirus disease 2019 (Covid-19), but definitive evidence is lacking.\n\nMethods: We conducted an open-label, cluster-randomized trial involving asymptomatic contacts of patients with polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia, Spain. We randomly assigned clusters of contacts to the hydroxychloroquine group (which received the drug at a dose of 800 mg once, followed by 400 mg daily for 6 days) or to the usual-care group (which received no specific therapy). The primary outcome was PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary outcome was SARS-CoV-2 infection, defined by symptoms compatible with Covid-19 or a positive PCR test regardless of symptoms. Adverse events were assessed for up to 28 days.\n\nResults: The analysis included 2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020. A total of 1116 contacts were randomly assigned to receive hydroxychloroquine and 1198 to receive usual care. Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]). In addition, hydroxychloroquine was not associated with a lower incidence of SARS-CoV-2 transmission than usual care (18.7% and 17.8%, respectively). The incidence of adverse events was higher in the hydroxychloroquine group than in the usual-care group (56.1% vs. 5.9%), but no treatment-related serious adverse events were reported.\n\nConclusions: Postexposure therapy with hydroxychloroquine did not prevent SARS-CoV-2 infection or symptomatic Covid-19 in healthy persons exposed to a PCR-positive case patient. (Funded by the crowdfunding campaign YoMeCorono and others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).
    """
    extract_PIO(test_sentence)
