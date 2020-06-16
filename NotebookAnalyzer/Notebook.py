import os
import sys
import json
import numpy as np
import pandas as pd
import functools
import re
import gc
from NotebookParser import NotebookParser
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


class Notebook():

    def __init__(self, type='json'):
        '''
        type: whether the loader is a pandas dataframe or a json object, defaults to json

        '''
        self.type = type
        self.content = None
        self.comment = None
        self.markdown = None
        self.package = None
        self.code = None
        self.code_block = None
        self.major_code_block = None

    def logging(func_name):
        def decorater(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                assert func_name in logging_dict.keys(), 'Unknown logging requirement'
                log_string = func.__name__+'running'
                if verbose:
                    print(log_string)
                    print(logging_dict[func_name])
                tmp = func(*args, **kw)

                if verbose:
                    print('finsing running '+func.__name__)

                return tmp
            return wrapper
        return decorater

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

    def load(self, data, verbose_mode=False):
        '''
        loads data into the object, defaults to json

        data: ipynotebook object, defaults to json

        verbose: turn on/off the verbose mode

        '''

        global verbose
        verbose = verbose_mode

        content = self._load_json(data)
        self._set_content(content)
        self.markdown = self._get_content()['markdown']
        self.code = self._get_content()['code']

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

    def get_header_structure(self, verbose_mode=False):

        data = self._to_csv(self.markdown)

        header_level = defaultdict(list)
        full_content = ''

        for i, row in data.iterrows():
            content = row['source']
            multi_lines = content.split('\n')
            for i, line in enumerate(multi_lines):

                if line.startswith('#'):
                    header_level[line.count('#')].append((row['id'], i))
                else:
                    line = re.sub(compilers['image'], re.sub(
                        compilers['url'], '', line), line)
                full_content += line + ';'

        return header_level, full_content

    def parse(self, verbose_mode=False):
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

    def get_code_block(self):

        # counts the number of code blocks
        block_counter = 0

        codes = self.code.split('\n')

        # all the code blocks in the code
        code_blocks = defaultdict(list)

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

                if prev >= hdpointer or prev <= tlpointer:

                    block_id = (prev, j-prev)
                    code_blocks[block_counter].append(
                        {block_id: '\n'.join(codes[prev:j])})
                    block_counter += 1
                    hdpointer = j
                    tlpointer = prev
                    indents[indent].remove(prev)

            indents[indent].append(j)
        self.code_block = code_blocks

    def _is_assignment(self, line):
        '''
        check if a line is a valid assignment
        the line must be a list of strings

        '''

        return '=' in line and line.index('=') < 2 and 'if' not in line

    def get_major_blocks(self):
        relevant_codes = []
        defs = []
        classes = []
        code_blocks = self.code_block
        edges = []

        for pocket in code_blocks.values():

            edges.extend([tup for p in pocket for tup in p.keys()])

        _ = edges.sort(key=lambda x: x[1], reverse=True)

        edge_set = set([edges[0][0], edges[0][0]+edges[0][1]])
        chunk = [edges[0]]

        for edg in edges[1:]:
            if edg[0] >= max(edge_set) or edg[0]+edg[1] <= min(edge_set):
                edge_set.update(set([edg[0], edg[0]+edg[1]]))

                chunk.append(edg)

        cs = self.code.split('\n')

        for start, range in chunk:

            if 'def' in cs[start]:
                defs.extend(list(np.arange(start, start+range)))

            elif 'class' in cs[start]:
                classes.extend(list(np.arange(start, start+range)))

            else:
                relevant_codes.extend(list(np.arange(start, start+range)))

        return relevant_codes, defs, classes

    def package_analyzer(self):
        '''
        extracting packages and track package use
        '''
        package_dicts = self.package
        aliases = list(package_dicts.keys())
        codes = self.code.split('\n')

        # An alias variable dictionary
        # A variable can be built upon multiple packages
        # A package can be used to build multiple variables

        alias_var = defaultdict(list)
        var_alias = defaultdict(list)

        # var_counter: records all the relevant things(either var or package) given a lhs variable
        # func_counter: records which line has a func

        var_counter = defaultdict(list)
        func_counter = defaultdict(bool)

        for i, line in enumerate(codes):

            line = re.sub('\t ', '', line)
            has_func = len(re.findall(compilers['func'], line))
            splitted = re.findall(compilers['splitter'], line)

            if not self._is_assignment(splitted):

                continue

            # separate the 'assigner' and the 'assignee'

            assign = splitted.index('=')
            var = ''.join(splitted[:assign])
            things = splitted[assign+1:]

            # check if the assignment is a function
            func_counter[i] = has_func != 0

            var_counter[var].append(i)

            for thing in things:
                # if the thing is a package
                if thing in aliases and has_func:
                    alias_var[thing].append(var)
                    var_alias[var].append(thing)
                    if has_func:
                        break

            # if the thing is already a variable
            for thing in things:
                if thing in var_counter.keys():
                    var_counter[thing].append(-i)

        return alias_var, var_alias, var_counter

    def full_analysis(self):
        '''

        Analysis of package usage


        '''

        alias_var, var_alias, var_counter = self.package_analyzer()

        alias_package_dict = self.package

        # stats

        # 0. package_string:所有package名称的合集

        package_string = ''

        for a, v in alias_var.items():

            vcounter = 0

            for var in v:
                vcounter += len(list(filter(lambda x: x <
                                            0, var_counter[var])))

            pack_name = alias_package_dict[a]
            package_string += ' '.join([pack_name]*(vcounter+1))+' '

        # 1. package_counts: notebook中出现的不同的包的个数

        package_counts = len(set(alias_var.keys()))

        # 2. package_variable_rate:平均每个package被多少个不同的variable征用

        package_variable_rate = np.mean(
            np.array([len(set(v)) for k, v in alias_var.items()]))

        # 3. variable_consumption_rate:每个variable被使用的次数
        variable_consumption_rate = np.mean(
            np.array([len(list(filter(lambda x:x < 0, v))) for k, v in var_counter.items()]))

        # 4. variable_update_rate:每个variable被update的次数
        variable_update_rate = np.mean(
            np.array([len(list(filter(lambda x:x > 0, v))) for k, v in var_counter.items()]))

        # 5. variable_package_rate:每个variable平均使用了多少个不同的package
        variable_package_rate = np.mean(
            np.array([len(set(v)) for k, v in var_alias.items()]))

        return {'package_str': package_string, 'package_counts': package_counts, 'package_variable_rate': package_variable_rate,
                'variable_consumption_rate': variable_consumption_rate,
                'variable_update_rate': variable_update_rate, 'variable_package_rate': variable_package_rate}

    def _major_lines(self, error=False):
        '''

        check if a line is in major block

        '''

        all_lines = self.code.split('\n')

        try:

            rel, defs, classes = self.get_major_blocks()
            return [i for i, line in enumerate(all_lines) if i not in defs and i not in classes]

        except IndexError:

            return [i for i, line in enumerate(all_lines)]

    def entity_tracing(self):

        codes = self.code
        data_tracker = defaultdict(list)
        root = Node('root')
        node_dictionary = {'root': root}
        valid_lines = self._major_lines()

        for i, line in enumerate(codes.split('\n')):

            if i not in valid_lines:

                continue

            line = re.sub(' ', '', line)

            has_data = len(re.findall(compilers['path'], line))
            splitted = re.findall(compilers['splitter'], line)

            if '=' in splitted and splitted.index('=') < 2 and 'if' not in splitted:

                assign = splitted.index('=')
                var = ''.join(splitted[:assign])
                things = splitted[assign+1:]

                if has_data:
                    data_tracker[var].append(i)
                    node_dictionary.update({var: Node(var, parent=root)})

                for thing in things:
                    if thing in node_dictionary.keys() and var != thing:
                        node_dictionary.update(
                            {var: Node(var, parent=node_dictionary[thing])})

        return node_dictionary

    def entity_analysis(self):

        node_dictionary = self.entity_tracing()

        root = node_dictionary['root']

        orders = [[node.name for node in children]
                  for children in LevelOrderGroupIter(root)]

        try:
            first_order_consumption = len(orders[1])
        except IndexError:
            first_order_consumption = 0

        try:
            second_order_consumption = len(orders[2])
        except IndexError:
            second_order_consumption = 0

        try:
            average_reference_length = np.mean(np.array(
                [grandchildren.height for child in root.children for grandchildren in child.children]))
        except:
            average_reference_length = 0
        prev = 0
        cur = 1
        level = 1
        max_expansion = 0
        while cur != prev:
            cur_level = [[node.name for node in children]
                         for children in LevelOrderGroupIter(root, maxlevel=level+1)][-1]
            prev_level = [[node.name for node in children]
                          for children in LevelOrderGroupIter(root, maxlevel=level)][-1]
            if (len(cur_level)/len(prev_level)) > max_expansion:
                max_expansion = len(cur_level)/len(prev_level)
            prev = len(prev_level)
            cur = len(cur_level)
            level += 1

        return {'first_order_consumption': first_order_consumption, 'second_order_consumption': second_order_consumption,
                'average_reference_length': average_reference_length, 'max_expansion': max_expansion}
