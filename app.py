from flask import Flask, render_template, request, url_for
from backend import Retrieve_PIO as rPIO
import os

template_dir = os.path.abspath('frontend/templates')
static_dir = os.path.abspath('frontend/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

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
    for source in output:
        target = {}
        target['Title'] = "Abstract title TBA"
        target['Content'] = "Content TBA"# source[0]
        target['Population'] = source[0]
        target['Intervention'] = source[1]
        target['Outcome'] = source[2]
        result.append(target)
    return result


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        population = request.form.get('population')
        others = request.form.get('others')
        output = rPIO.get_PIO(population, others)
        return render_template('search.html', abstracts=convert_PIO(output), search=[population, others])

if __name__ == "__main__":
    app.run(debug=True)
