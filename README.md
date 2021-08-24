# pediatrics-decision-support
The codes in "backend" folder need corresponding datasets and model weights. One can download these datasets and model weights from https://wcm.box.com/s/xxairqjh697wlf0l0tcd9oqkj5fr1dlj. Just put them into the "backend" folder and the codes will work.

In the "backend" folder, there are four python files.
1. "Retrieval.py" corresponds to the retreival module. It will select the most relevant abstracts according to the query.
2. "PIO_classification.py" belongs to the PIO classification module. It will give each word a label (whether it is Population, Intervention, Outcome or nothing).
3. "extract_PIO_elements.py" belongs to the PIO classification module too. It will extract or summarize the PIO elments according to the labels provided by "PIO_classification.py" by eliminating duplicates and noises.
4. "PIO_training.py" is used for training the PIO classifier which is used in "PIO_classification.py" and "extract_PIO_elements.py". We do not need run it once we get the model.

In the "frontend" folder, there are two folders. One is the "static" folder including the .css files for the frontend. The other is the "Templates" folder including the .html files for the frontend.

"app.py" is a demo of flask. Ali and Gur can start with this demo to manage or organize all the frontend and backend modules.

# Dependancies

 - transformers
 - torch
 - Levenshtein
