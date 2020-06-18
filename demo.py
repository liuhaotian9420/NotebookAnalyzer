from NotebookAnalyzer import Notebook
from NotebookAnalyzer import NotebookAnalyzer
from NotebookAnalyzer import NotebookStats  as ns
import pandas as pd
import json
import os
import sys

test_json = pd.read_csv('./tests/gzlt_team_with_score_members_public_labs_data.csv')
test_nb = test_json.query('lab_id=="5d40ff29c143cf002bcc3721"')['ipynb_data'].values[0]



def main():
    nb = Notebook.Notebook()
    nb = nb.read_notebook(test_nb)

    analyzer = NotebookAnalyzer.NotebookAnalyzer()
    analyzer.load(nb)
    codes = analyzer.analyze('code')[0]

    package_analyzer = NotebookAnalyzer.NotebookAnalyzer()
    package_analyzer.load(nb)
    pks = package_analyzer.analyze('package')[0]

    for i, pk in enumerate(pks):

        ns.full_analysis(pk[0],pk[1],codes[i][1],nb.package)
        



if __name__ == '__main__':

    main()

