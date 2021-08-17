from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    population = request.form.get('population')
    intervention = request.form.get('intervention')
    return render_template('search.html')

if __name__ == "__main__":
    app.run(debug=True)