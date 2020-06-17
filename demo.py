from NotebookAnalyzer import Notebook
import pandas as pd
import json
import os
import sys

test_json = pd.read_csv('./tests/gzlt_team_with_score_members_public_labs_data.csv')
test_nb = test_json.query('lab_id=="5d40ff29c143cf002bcc3721"')['ipynb_data'].values[0]



def main():
    nb = Notebook.Notebook()
    nb = nb.read_notebook(test_nb)
    



if __name__ == '__main__':

    main()

