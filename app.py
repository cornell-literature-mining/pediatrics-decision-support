from flask import Flask, render_template, request, url_for
from backend import Retrieve_PIO as rPIO
import os
import json

template_dir = os.path.abspath('frontend/templates')
static_dir = os.path.abspath('frontend/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


# Because it runs extremely slowly on my computer


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

def convert_PIO(output):
    '''
    Convert to format: List<Dict<string, string>>, where dict has:
        - Title: the title of the abstract
        - Content: the content
        - Population, Intervention, Outcome
    '''
    result = []
    for i in range(len(output[0])):
        target = {}
        pio = output[0][i]
        data = output[1][i]
        target['Title'] = data[0]
        target['Authors'] = data[1]
        target['Date'] = data[2]
        target['FullAbstract'] = data[3]
        target['PMID'] = data[4]
        target['Population'] = pio[0]
        target['Intervention'] = pio[1]
        target['Outcome'] = pio[2]
        result.append(target)
    return result


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        population = request.form.get('population')
        others = request.form.get('others')
        output = rPIO.get_PIO(population, others)
        print("\n\n=== OUTPUT ===\n\n" + str(output))
        return render_template('search.html', abstracts=convert_PIO(output), search=[population, others])
    else:
        return render_template('home.html')


@app.route('/view', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        if 'view' in request.form:
            abstract = request.form.get('view')
            abstract = json.loads(abstract.replace("'", '"'))
            print(abstract)
            return render_template('view.html', abstract=abstract, test="Hello world!")
        else:
            population = request.form.get('population')
            others = request.form.get('others')
            output = TEMP_OUTPUT# rPIO.get_PIO(population, others)
            print("\n\n=== OUTPUT ===\n\n" + str(output))
            return render_template('search.html', abstracts=convert_PIO(output), search=[population, others])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
