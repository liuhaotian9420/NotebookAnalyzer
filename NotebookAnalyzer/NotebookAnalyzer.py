import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from Notebook import Notebook
from utils.utils import get_compiler
from collections.abc import Iterable
from anytree import Node, RenderTree, LevelOrderGroupIter


class Analyzer():

    def __init__(self, subject: Notebook):

        self.notebook = subject

    def __repr__(self):

        return 'An analyzer'

    def _code_analysis(self):
        '''

        track the evolution of variable/data entities in the code


        '''

        data_tracker = defaultdict(list)

        root = Node('root')
        node_dictionary = {'root': root}
        path_compiler = get_compiler('path')
        for var, things, lineNum in self.get_assignment_lines(with_linenum=True).values():
            has_data = len(re.findall(path_compiler, ''.join(things)))
            if has_data:
                data_tracker[var].append(lineNum)
                node_dictionary.update({var: Node(var, parent=root)})
            for thing in things:
                if thing in node_dictionary.keys() and var != thing:
                    node_dictionary.update(
                        {var: Node(var, parent=node_dictionary[thing])})

        return node_dictionary

    def _code_block_analysis(self, with_code=True):
        '''

        get major block and returns the block titles

        code_blocks is a list of tuples

        '''
        major_block_dict = defaultdict(tuple)
        code_blocks = self.notebook.code_block
        assert isinstance(
            code_blocks, list), 'code_blocks must be a list of tuples'
        _ = code_blocks.sort(key=lambda x: x[1], reverse=True)
        edges = set([code_blocks[0][0], code_blocks[0][0]+code_blocks[0][1]])
        major_block = [edges]
        # reversely search for the largest edges
        for edge in code_blocks[1:]:
            if edge[0] >= max(edges) or edge[0]+edge[1] < min(edges):
                new_block = set(edge[0], edge[0]+edge[1])
                edges.update(new_block)
                major_block.append(new_block)

        codes = self.notebook.code.split('\n')

        for start, end in major_block:
            major_block_dict[start] = (end, codes[start:end])

        if with_code:
            for start, end in major_block:
                major_block_dict[start] = (end, codes[start:end])
        else:
            major_block_dict = major_block

        return major_block_dict

    def _package_analysis(self):
        '''
        extracting packages and track package use

        '''
        package_dicts = self.notebook.package
        aliases = list(package_dicts.keys())

        alias_var = defaultdict(list)

        func_counter = defaultdict(bool)

        func_compiler = get_compiler('func')

        for var, things, lineNum in self.get_assignment_lines(with_linenum=True).values():

            line = var+''.join(things)

            has_func = len(re.findall(func_compiler, line))

            func_counter[lineNum] = has_func != 0

            # if there is a function,only consider the first alias

            for thing in things:
                if thing in aliases:
                    alias_var[thing].append(var)
                    if has_func:
                        break

        return alias_var

    def _markdown_analysis(self):
        '''
        Line content always follow the previous header

        '''

        header_level = defaultdict(list)
        header_content = defaultdict(list)
        current_header = ''
        for lineNum, line in enumerate(self.notebook.markdown):

            if line.startswith('#'):

                header_level[line.count('#')].append(lineNum)
                current_header = line

            else:

                line.re.sub(get_compiler('image'), re.sub(
                    get_compiler('url'), '', line), line)
                header_content[current_header].append(line)

        return header_level, header_content

    def _variable_analysis(self, with_package=False):
        '''
        track package usage

        '''

        var_alias = defaultdict(list)
        var_counter = defaultdict(list)
        package_dicts = self.notebook.package

        for lineNum, t in self.get_assignment_lines().items():

            var_counter[t[0]].append(lineNum)

            for thing in t[1]:

                if thing in var_counter.keys():
                    var_counter[thing].append(-lineNum)

                if thing in package_dicts.keys():

                    var_alias[t[0]].append(thing)

        return var_alias, var_counter

    def _is_assignment(self, line):

        return '=' in line and line.index('=') < 2 and 'if' not in line

    def get_assignment_lines(self, with_linenum=False):
        '''

        return list of (var,things)

        '''
        split_compiler = get_compiler('splitter')
        result = defaultdict(tuple)
        for lineNum, line in enumerate(self.notebook.code.split('\n')):

            line = re.sub('\t', '', line)

            splitted = re.findall(split_compiler, line)

            if not self._is_assignment(splitted):

                continue
            assign = splitted.index('=')
            var = ''.join(splitted[:assign])
            things = splitted[assign+1:]  # things on rhs of tre assignment
            if not with_linenum:
                result[lineNum] = (var, things)
            else:
                result[lineNum] = (var, things, lineNum)

        return result


class NotebookAnalyzer():

    def __init__(self):

        self.analyzers = None

    def _load(self, input):
        '''
        import can be a list of notebooks or a single notebook


        '''
        assert isinstance(input, Notebook) or isinstance(
            input, Iterable), 'invalid input type'

        if isinstance(input, Iterable):
            for i in input:
                assert isinstance(i, Notebook), 'invalid input type'
        else:
            input = [input]

        return input

    def analyze(self, input, analyzers):
        '''
        analyzers:
        - names for analyzers,list-like
        - use 'all' all analyzers are applied
        '''
        assert analyzers == 'all' or isinstance(
            analyzers, Iterable), 'invalid argument for analyzers'

        analyzers_name = ['code', 'code_block',
                          'variable', 'package', 'markdown']
        if analyzers == 'all':

            self.analyzers = analyzers_name

        elif set(analyzers)-set(analyzers_name):

            raise Exception('Invalid analyzer name')
        else:
            self.analyzers = analyzers

        nb_data = []

        for notebook in self._load(input):

            analyzer_model = Analyzer(notebook)
            analyzer_result = {}

            for analyzer in self.analyzers:

                func_name = '_'+analyzer+'_analysis'
                analyzer_result.update(
                    {analyzer: getattr(analyzer_model, func_name)})
            nb_data.append(analyzer_result)
