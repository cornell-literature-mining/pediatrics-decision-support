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

def get_PIO(population, others):
    # TODO: include the abstract itself (including the title too would be good as well) in addition to the PIO lists
    # Assumed format: lsit of "[Content (string), P list (list), I list (list), O list (list)]

    # Commenting out as it doesn't work on my PC:
    #retrieval_input = create_ret_input(population, others)
    #selected_abstracts = ret.retrieval(retrieval_input)
    #output_PIO = PIO(selected_abstracts)

    # Temp output - delete once the NVIDIA GPU problem is sorted out
    output_PIO = [["Humans need food and water in order to stay healthy.", ["Humans"], ["Food", "Water"], ["Healthy"]], ["Cats need pets in order to stay happy. This is a different content.", ["Cats"], ["Pets"], ["Happy"]]]
    return output_PIO
