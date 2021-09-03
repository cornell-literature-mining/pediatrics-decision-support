import json
from . import Retrieval as ret
from . import extract_PIO_elements as xPIO
from . import File_loading as fl

def create_ret_input(population, others):
    pop_tag = "P"
    Q = pop_tag + " " + population + " " + pop_tag + " " + others
    #print(Q)
    with fl.Open('PIO_data_PMID_abstracts.json', 'r') as f:
        PMID_abstracts = json.load(f)
    size = 57
    abstracts = [0 for element in range(size)]
    PMID = [0 for element in range(size)]
    for i in range(size):
      abstracts[i] = PMID_abstracts[i][1]
      PMID[i] = PMID_abstracts[i][0]
    retrieval_input_p1 = [0 for element in range(size)]
    for i in range(size):
        retrieval_input_p1[i] = [Q, abstracts[i]]
    retrieval_input = [retrieval_input_p1,PMID]
    return retrieval_input

def PIO(selected_abstracts):
    size = len(selected_abstracts)
    output_PIO = [0 for element in range(size)]
    for i in range(size):
        P_list, I_list, O_list = xPIO.extract_PIO(selected_abstracts[i])
        output_PIO[i] = [P_list, I_list, O_list]
        #print(output_PIO[i])
    return output_PIO

def find_entry(pmid):
    with fl.Open('database_info.json', 'r') as json_file:
        database_info = json.load(json_file)

    size_database = len(database_info)
    index = 0

    for j in range(size_database):
        if pmid == database_info [j][4]:
            print("found match")
            index = j
            break

    return database_info[index]

def find_info(pmids_selected_abstracts):
    size = len(pmids_selected_abstracts)
    info = []
    for i in range(size):
        info.append(find_entry(pmids_selected_abstracts[i]))
    return info

def update_info(selected_abstracts_info,selected_abstracts):
    size = len(selected_abstracts)
    for i in range(size):
        selected_abstracts_info[i][3] = selected_abstracts[i]
    return selected_abstracts_info

def get_PIO(population, others):
    retrieval_input = create_ret_input(population, others)
    selected_abstracts,selected_pmids = ret.retrieval(retrieval_input)
    #print("there are:",len(selected_abstracts),"selected abstracts")
    #print("there are:",len(selected_pmids),"selected pmids")
    selected_abstracts_info = find_info(selected_pmids) #[title,authors_list,year,abstract,pmid]
    updated_selected_abstracts_info = update_info(selected_abstracts_info,selected_abstracts)
    #print("there are:",len(selected_abstracts_info),"selected infos")
    output_PIO = PIO(selected_abstracts)
    return output_PIO,updated_selected_abstracts_info
