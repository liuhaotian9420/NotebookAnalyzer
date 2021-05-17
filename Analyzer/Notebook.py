import os
import sys
import json
import numpy as np
import pandas as pd
import functools
import re
import gc
from .NotebookParser import NotebookParser
from .NotebookCode import Code, ClassCode
from collections import Counter, defaultdict
from anytree import Node, RenderTree, LevelOrderGroupIter

compilers = {

    'url': re.compile(r'\(http.+?\)', re.DOTALL),
    'image': re.compile(r'!\[.+?\]', re.DOTALL),
    'package_alias': re.compile(r'(?<=import\s)[\w+\s*,*]+(?=\sas\s)|(?<=\sas\s)[\w+,*]+'),
    'tq': re.compile(r'(?<=""")[^"""]+(?=""")'),
    'dq': re.compile(r'(?<=\'\'\')[ ^\'\'\']+(?=\'\'\')'),
    'ns': re.compile(r'(?<=#)[^\n\r]+(?=\n)|(?<=#)[^\n\r]+(?=\r)'),
    'package': re.compile(r'(?<=import\s)[\w+\s*,*]+|(?<=\sas\s)[\w+,*]+'),
    'path': re.compile(r'(?<=\').*/\w*\d*'),
    'entity': re.compile(re.compile(r'.*?=')),
    'func': re.compile(r'\.*\w+\d*\(.*\)'),
    'splitter': re.compile(r'\w+|='),
    'value': re.compile(r'=.*'),
    'tabspace': re.compile(r'^[\s|\t]+\w*?')}

logging_dict = {

    '_load_json': 'loading notebooks json object',
    '_to_csv': 'converting from json objects into csv'


}


def logging(func_name):
    def decorater(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kw):
            assert func_name in logging_dict.keys(), 'Unknown logging requirement'
            log_string = func.__name__+'running'
            if verbose:
                print(log_string)
                print(logging_dict[func_name])
            tmp = func(self,*args, **kw)

            if verbose:
                print('finsing running '+func.__name__)

            return tmp
        return wrapper
    return decorater


class Notebook():

    def __init__(self, type='json',content = None):
        '''
        type: whether the loader is a pandas dataframe or a json object, defaults to json
        content: when creating a Notebook, initialize with some content

        '''
        self.type = type

        if not content:

            self.content = content

        self.parsed = {}
        
        self.comment = None
        self.markdown = None
        self.package = None
        self.code = None  # now contains all the original code information
        self.major_block = None
        self.code_list = []

    @logging('_load_json')
    def _load_json(self, data):
        '''

        json_loading  of notebook data

        '''
        try:
            jdata = json.loads(data)

        except:
            raise Exception('Cannot read json data')

        assert 'cells' in jdata.keys(), 'No cells found in notebook'

        content = defaultdict(list)

        for j in jdata['cells']:

            content[j['cell_type']].append(j)

        return content

    def _load(self, data, verbose_mode=False):

        '''
        loads data into the object, defaults to json

        data: ipynotebook object, defaults to json

        verbose: turn on/off the verbose mode

        '''

        global verbose
        verbose = verbose_mode

        if self.type == 'json':
            try:
                content = self._load_json(data)
            except:
                raise Exception('Notebook requires json files')

        self._set_content(content)
        self.markdown = self._get_content()['markdown']
        self.code = self._get_content()['code']

        return self

    def read_notebook(self, data, pipe='all',verbose_mode =False):
        
        '''
        data: external data, must match the type of notebook
        pipe: pipeline for data processsing, can be a series of strings

        '''

        nb = self._load(data)
        if pipe =='all':
            nb = nb._parse()._get_code_block()._code_block_analysis()

        return nb


    @logging('_to_csv')
    def _to_csv(self, content):
        '''
        convert object from json into csv and create unique row_id

        '''
        # assert self.content not None,'there is no content in the notebook'

        cdata = pd.DataFrame.from_dict(content)
        cdata['id'] = cdata.apply(lambda x: x['metadata']['id']
                                  if 'id' in x['metadata'].keys() else '00000', axis=1)

        return cdata

    def _get_content(self):

        return self.content

    def _set_content(self, content):

        self.content = content

    def get_code(self):

        return self.code

    def _parse(self, verbose_mode=False):
        '''
        read and parse the code into different sections:

        1. comments and docstrings
        2. imports and packages
        3. code
        4. code blocks

        returns a dictionary containing different sections of code

        '''
        data = self._to_csv(self.code)

        # packages
        package_dicts = {}

        # identify cell range
        block_tracker = defaultdict(tuple)

        # all kinds of comments
        comments = defaultdict(list)

        # all the codes from different rows
        cs = ''

        comment_parser = NotebookParser(compilers=compilers)

        line_counter = 0

        for i, row in data.iterrows():

            code = row['source']  # 部分source数据会出现list type

            block_start = line_counter

            cmt, text = comment_parser.parse_comment(code, comment_type='ns')

            dq, text = comment_parser.parse_comment(text, comment_type='dq')

            tq, text = comment_parser.parse_comment(text, comment_type='tq')

            comments[row['id']].append((cmt, dq, tq))

            for line in text.split('\n'):
                line_counter += 1
                if line == '':
                    continue
                if '#' in line:
                    continue
                cs += line+'\n'

                if line.startswith('import ') or line.startswith('from '):
                    package_dicts.update(comment_parser.parse_package(line))

            block_end = line_counter
            block_tracker[row['id']] = (block_start, block_end)

        self.code = cs
        self.package = package_dicts
        self.comment = comments

        return self

    def _get_code_block(self,pipe = True):

        # counts the number of code blocks
        block_counter = 0

        codes = self.code.split('\n')

        # all the code blocks in the code
        code_blocks = []

        indents = defaultdict(list)
        indent_array = []
        hdpointer = 0
        tlpointer = len(codes)

        for j, line in enumerate(codes):

            

            try:
                s = re.findall(compilers['tabspace'], line)[0]
                indent = int(s.count('\t')+s.count(' ')/4)
            except:
                indent = 0

            indent_array.append(indent)
            if indents.get(indent, None) != None and indents.get(indent, None)[-1]+1 != j:
                prev = indents.get(indent, None)[-1]

                if (prev >= hdpointer or prev <= tlpointer) and all([i>=indent for i in indent_array[prev:j]]):

                    block_id = (prev, j - prev)
                    code_blocks.append(block_id)
                    block_counter += 1
                    hdpointer = j
                    tlpointer = prev
                    indents[indent].remove(prev)

            indents[indent].append(j)

        if pipe:

            return self

        else:

            return code_blocks

    def _code_block_analysis(self):
        '''

        get major block and returns the block titles

        code_blocks is a list of tuples

        '''

        code_blocks = self._get_code_block(pipe=False)
        assert isinstance(
            code_blocks, list), 'code_blocks must be a list of tuples'
        _ = code_blocks.sort(key=lambda x: x[1], reverse=True)
        major_block = []
        current_range = []

        for edge in code_blocks:
            if not self._in_current_range(edge, current_range):
                current_range = self._add_range(current_range, edge)
                major_block.append((edge[0],edge[0]+edge[1]))

        self.major_block = major_block
        codes = self.code.split('\n')
        block_codes = []
        block_lines = []

        for start, end in major_block:

            block_lines.extend(list(np.arange(start, end)))

            line = re.sub('\t\(\),', '', codes[start]).split(' ')

            if 'class' in line or 'def' in line:

                block_type = line[0]
                block_name = line[1]
                params = ' '.join(line[1:])
            else:
                block_name = line[0]
                block_type = block_name
                params = ''

            block_codes.append(
                ClassCode('\n'.join(codes[start:end]), block_type,block_name, params))



        # re-define code
        self.code_list.append(Code('\n'.join(
            [line for idx, line in enumerate(self.code.split('\n')) if idx not in block_lines])))
        
        self.code_list.extend(block_codes)

        return self
    
    def _in_current_range(self,tup, current_range):
        t_range = np.arange(tup[0], tup[0]+tup[1])
        return len(set(t_range).intersection(set(current_range))) != 0

    def _add_range(self, current_range, tup):

        cr = current_range
        cr.extend(list(np.arange(tup[0],tup[0]+tup[1])))
        
        return cr

    
        

