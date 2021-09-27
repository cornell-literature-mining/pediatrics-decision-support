from flask import Flask, render_template, request, url_for
from backend import Retrieve_PIO as rPIO
import os
import json
import difflib

# Load JSONs
file = open("desc2020.json", "r", encoding="cp866")
descs = json.loads("[" + file.read().replace("\n", "\n,") + "]") # Currently only using desc as I have no idea what each one is
file.close()
MeSH = []
for desc in descs:
    MeSH.append(desc["name"])
# Init app
template_dir = 'frontend/Templates'
static_dir = 'frontend/static'
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Because it runs extremely slowly on my computer
TEMP_OUTPUT = ([[['children with other psychiatric disorders (e', 'normal children (n = 70)', 'normal children and children with pervasive developmental disorders.', 'sample of normal children (n = 12)', 'children with Attention-deficit Hyperactivity Disorder ; n = 32)', 'of children with PDDs (n = 10)', 'children with PDDs (n = 20)', 'normal children and children with pervasive developmental disorders (PDDs).', 'young children'], ['TOM test', '[CLS]', 'TOM'], ['internal consistency', 'scores', 'Intelligence Scale', 'intelligence', 'TOM test scores.', 'reliability and validity', 'interrater reliability', 'test-retest stability']], [['clinical model of acute allergic conjunctivitis.'], ['placebo', 'placebo. Olopatadine', 'ketorolac ophthalmic solution', 'Olopatadine', 'ketorolac', 'olopatadine ophthalmic solution'], ['hyperemia in conjunctival, ciliary, and episcleral vessel beds.', 'efficacy and safety', 'hyperemia', 'ocular itching and hyperemia', 'hyperemia and ocular itching', 'itching', 'comfortable', 'ocular itching']], [['patients with clinically organ-confined disease.', '114 patients', '259 men with prostate cancer underwent', 'either for three', 'clinical stage B whereas 41', 'tumours', 'organ-confined untreated prostate cancer (PCa)', 'patients with prostate cancer and treated with', 'in the 26 centres participating in the Italian randomised prospective PROSIT study.', 'prostate cancer.', 'clinical stage C.', '(RP)'], ['" neoadjuvant " hormonal therapy (NHT).', 'total androgen ablation', 'neoadjuvant therapy', 'RP.', 'radical prostatectomy', 'total androgen ablation (e', 'NHT', 'radical retropubic prostatectomy and bilateral pelvic node dissection', 'NHT therapy'], ['pathologic stage of PCa and resection limit status', 'resection limit status', 'cancers with negative margins', 'negative margins', 'pathological stage B', 'and eosin', 'stage', 'pathologic stage and resection limit status']]], [['The TOM test: a new instrument for assessing theory of mind in normal children and children with pervasive developmental disorders.', ['Muris P', 'Steerneman P', 'Meesters C', 'Merckelbach H', 'Horselenberg R', 'van den Hogen T', 'van Dongen L'], '1999', 'The TOM test: a new instrument for assessing theory of mind in normal children and children with pervasive developmental disorders.\n\nThis article describes a first attempt to investigate the reliability and validity of the TOM test, a new instrument for assessing theory of mind ability in normal children and children with pervasive developmental disorders (PDDs). In Study 1, TOM test scores of normal children (n = 70) correlated positively with their performance on other theory of mind tasks. Furthermore, young children only succeeded on TOM items that tap the basic domains of theory of mind (e.g., emotion recognition), whereas older children also passed items that measure the more mature areas of theory of mind (e.g., understanding of humor, understanding of second-order beliefs). Taken together, the findings of Study 1 suggest that the TOM test is a valid measure. Study 2 showed for a separate sample of normal children (n = 12) that the TOM test possesses sufficient test-retest stability. Study 3 demonstrated for a sample of children with PDDs (n = 10) that the interrater reliability of the TOM test is good. Study 4 found that children with PDDs (n = 20) had significantly lower TOM test scores than children with other psychiatric disorders (e.g., children with Attention-deficit Hyperactivity Disorder; n = 32), a finding that underlines the discriminant validity of the TOM test. Furthermore, Study 4 showed that intelligence as indexed by the Wechsler Intelligence Scale for Children was positively associated with TOM test scores. Finally, in all studies, the TOM test was found to be reliable in terms of internal consistency. Altogether, results indicate that the TOM test is a reliable and valid instrument that can be employed to measure various aspects of theory of mind.\n\n', '10097996'], ['Comparative evaluation of olopatadine ophthalmic solution (0.1%) versus ketorolac ophthalmic solution (0.5%) using the provocative antigen challenge model.', ['Deschenes J', 'Discepola M', 'Abelson M'], '1999', 'Comparative evaluation of olopatadine ophthalmic solution (0.1%) versus ketorolac ophthalmic solution (0.5%) using the provocative antigen challenge model.\n\nOBJECTIVE This study was conducted to compare the efficacy and safety of olopatadine ophthalmic solution (0.1%) with ketorolac ophthalmic solution (0.5%) in a clinical model of acute allergic conjunctivitis. Olopatadine is a dual acting H1 histamine receptor antagonist and a mast cell stabilizer, shown to be effective in treating allergic conjunctivitis. Ketorolac is a non-steroidal anti-inflammatory drug approved in the United States for the relief of ocular itching associated with seasonal allergic conjunctivitis.\nMETHODS The provocative antigen challenge model was used in this randomized, double-blind, single-center, crossover study. The allergen and concentration that consistently elicited a positive allergic reaction was used for challenge. After at least 14 days, subjects were randomized to receive either olopatadine in one eye and placebo in the contralateral eye, or ketorolac in one eye and placebo in the contralateral eye. Twenty-seven minutes after drug instillation subjects were challenged with allergen. At 3, 10, and 20 minutes following allergen challenge, subjects graded ocular itching and were assessed for hyperemia in conjunctival, ciliary, and episcleral vessel beds. Approximately 14 days later, subjects received the alternate treatment in one eye and placebo in the contralateral eye. They were again challenged with allergen and their responses were rated in the same manner.\nRESULTS Olopatadine significantly (p < 0.0001) reduced both ocular itching and hyperemia in all three vessel beds compared to placebo at all time points tested following allergen challenge. Ketorolac did not significantly reduce itching and showed a trend of increased hyperemia compared to placebo. Olopatadine was significantly (p < 0.001) more effective than ketorolac in reducing hyperemia and ocular itching at all time points and was also significantly (p < 0.05) more comfortable than ketorolac as reported by subjects immediately following drug instillation.\nCONCLUSION The study demonstrated that olopatadine is effective and safe in preventing and treating ocular itching and hyperemia associated with acute allergic conjunctivitis and is more effective and more comfortable than ketorolac.\n\n', '10337433'], ['Effect of total androgen ablation on pathologic stage and resection limit status of prostate cancer. Initial results of the Italian PROSIT study.', ['Montironi R', 'Diamanti L', 'Santinelli A', 'Galetti-Prayer T', 'Zattoni F', 'Selvaggi FP', 'Pagano F', 'Bono AV'], '1999', 'Effect of total androgen ablation on pathologic stage and resection limit status of prostate cancer. Initial results of the Italian PROSIT study.\n\nThe likelihood of finding organ-confined untreated prostate cancer (PCa) by pathological examination at the time of radical prostatectomy (RP) is only 50% in patients with clinically organ-confined disease. In addition, tumour is present at the resection margin in approximately 30% of clinical T2 (clinical stage B) cases. The issue of clinical "understaging" and of resection limit positivity have led to the development of novel management practices, including "neoadjuvant" hormonal therapy (NHT). The optimal duration of NHT is unknown. We undertook the present analysis to evaluate the effect of NHT on pathologic stage of PCa and resection limit status in patients with prostate cancer and treated with total androgen ablation either for three or six months before RP. Between January 1996 and February 1998, 259 men with prostate cancer underwent radical retropubic prostatectomy and bilateral pelvic node dissection in the 26 centres participating in the Italian randomised prospective PROSIT study. Whole mount sectioning of the complete RP specimens was adopted in each centre for accurately evaluating the pathologic stage and resection limit status. By February 1998, haematoxylin and eosin stained sections from 155 RP specimens had been received and evaluated by the reviewing pathologist (RM). 64 cases had not been treated with total androgen ablation (e.g. NHT) before RP was performed, whereas 58 and 33 had been treated for three and six months, respectively. 114 patients were clinical stage B whereas 41 were clinical stage C. After three months of total androgen ablation, pathological stage B was more prevalent among patients with clinical B tumours, compared with untreated patients (57% in treated patients vs. 36% in untreated). The percentage of cancers with negative margins was statistically significantly greater in patients treated with neoadjuvant therapy than those treated with immediate surgery alone (69% vs. 42%, respectively). After six months of NHT therapy the proportion of patients with pathological stage B (67% vs. 36%, respectively) and negative margins was greater than after 3 months (92% vs. 42%, respectively). For clinical C tumours, the prevalence of pathological stage B and negative margins in the patients treated for either 3 or 6 months was not as high as in the clinical B tumours, when compared with the untreated group (pathological stage B: 31% and 33% vs. 6% in the clinical C cases, respectively. Negative margins: 56% and 67% vs. 31%, respectively). The initial results of this study suggest that total androgen ablation before RP is beneficial in men with clinical stage B because of the significant pathological downstaging and decrease in the number of positive margins in the RP specimens. These two effects are more pronounced after six months of NHT than after three months of therapy. The same degree of beneficial effects are not observed in clinical C tumours.\n\n', '10337657']])

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
        output = rPIO.get_PIO_contrastive(population, others) # TEMP_OUTPUT #
        print("\n\n=== OUTPUT ===\n\n" + str(output))
        output = convert_PIO(output)
        # Find similar words in MeSH
        similar = difflib.get_close_matches(population + " " + others, MeSH, 4)
        return render_template('search.html', abstracts=output, search=[population, others], similar=similar)
    else:
        return render_template('home.html')

@app.route('/search2', methods=['GET', 'POST'])
def search2():
    if True:# request.method == 'POST':
        population = request.form.get('population')
        others = request.form.get('others')
        output = rPIO.get_PIO_contrastive(population, others)
        print("\n\n=== OUTPUT ===\n\n" + str(output))
        output = convert_PIO(output)
        output2 = {}
        for abstract in output:
            for out in abstract['Outcome']:
                if out in output2.keys():
                    output2[out].append(abstract)
                else:
                    output2[out] = [abstract]
        return render_template('search2.html', factors=output2, search=[population, others])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
