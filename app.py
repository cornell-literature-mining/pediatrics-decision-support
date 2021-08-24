from flask import Flask, render_template, request, url_for
from backend import Retrieve_PIO as rPIO

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
	if request.method == 'POST':
		population = request.form.get('population')
		others = request.form.get('others')
		output = rPIO.get_PIO(population,others)
		return render_template('search.html')

if __name__ == "__main__":
	app.run(debug=True)
