import json
from . import Retrieval as ret
from . import extract_PIO_elements as xPIO
from . import File_loading as fl

def create_ret_input(population, others):
    pop_tag = "P"
    Q = pop_tag + " " + population + " " + pop_tag + " " + others
    #print(Q)
    with fl.Open('test_abstracts.json', 'r') as json_file:
        database_abstracts = json.load(json_file)
    size = len(database_abstracts)
    retrieval_input = [0 for element in range(size)]
    for i in range(size):
        retrieval_input[i] = [Q, database_abstracts[i]]
    return retrieval_input

def PIO(selected_abstracts):
    size = len(selected_abstracts)
    output_PIO = [0 for element in range(size)]
    for i in range(size):
        P_list, I_list, O_list = xPIO.extract_PIO(selected_abstracts[i])
        output_PIO[i] = [P_list, I_list, O_list]
        #print(output_PIO[i])
    return output_PIO

def find_entry(selected_abstract):
    with fl.Open('database_info.json', 'r') as json_file:
        database_info = json.load(json_file)

    with fl.Open('PIO_data_PMID_abstracts.json', 'r') as json_file:
        database_abstracts = json.load(json_file)

    size_database = len(database_abstracts)

    for j in range(size_database):
        if selected_abstract == database_abstracts[j]:
            print("found match")
            index = j
            break

    return database_info[j]

def find_info(selected_abstracts):
    size = len(selected_abstracts)
    selected_abstracts_info = [0 for element in range(size)]
    info = []
    for i in range(size):
        info.append(find_entry(selected_abstracts[i]))
    return info


def get_PIO(population, others):
    retrieval_input = create_ret_input(population, others)
    selected_abstracts = ret.retrieval(retrieval_input)
    #print("there are:",len(selected_abstracts),"selected abstracts")
    selected_abstracts_info = find_info(selected_abstracts) #[title,authors_list,year,abstract,pmid]
    #print("there are:",len(selected_abstracts_info),"selected infos")
    print(selected_abstracts_info[0])
    output_PIO = PIO(selected_abstracts)
    return output_PIO,selected_abstracts_info

#get_PIO("men", "health")
