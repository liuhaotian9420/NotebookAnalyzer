import unittest
import pandas as pd
import json
import sys
import os
sys.path.append('../NotebookAnalyzer/')
from Notebook import Notebook
print(os.getcwd())

test_file_single = pd.read_csv('./gzlt_team_with_score_members_public_labs_data.csv')
test_file_array = pd.read_csv('./gzlt_team_with_score_members_private_labs_data_20200603.csv')

test_json = test_file_single.query('lab_id=="5d40ff29c143cf002bcc3721"')['ipynb_data'].values[0]


class NotebookTest(unittest.TestCase):
    def setUp(self):
        print('创建测试用的Notebook……')
        self.notebook = Notebook()

    def test_load(self):
        print('测试Notebook数据的读取')
        minute_nb = '{"cells":[{"cell_type":"markdown","source":"1"},{"cell_type":"code","source":"2"}]}'
        print(minute_nb)
        content = {'markdown':[{'cell_type':'markdown','source':'1'}],'code':[{'cell_type':'code','source':'2'}]}
        self.notebook.load(minute_nb)
        self.assertEqual(self.notebook.content, content)
        self.assertEqual(self.notebook.code,[{'cell_type':'code','source':'2'}])
        self.assertEqual(self.notebook.markdown,[{'cell_type':'markdown','source':'1'}])


if __name__ == '__main__':
    unittest.main()
