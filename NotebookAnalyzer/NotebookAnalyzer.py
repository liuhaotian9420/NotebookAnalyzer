import numpy as np
import pandas as pd
import gc
import re
from collections import Counter, defaultdict
from .Notebook import Notebook
from .NotebookCode import Code,ClassCode
from utils.utils import get_compiler
from collections.abc import Iterable
from anytree import Node, RenderTree, LevelOrderGroupIter


class Analyzer():

    def __init__(self, content):
        
        '''

        content: to be analyzed

        '''

        self.content = content

    def __repr__(self):

        return self.content

    def _is_assignment(self, line):

        return '=' in line and line.index('=') < 2 and 'if' not in line

    
    def get_assignment_lines(self,keep_original=True):
        '''

        return list of (var,things)

        '''
        split_compiler = get_compiler('splitter')
        result = defaultdict(tuple)
        code = self.content
            
        for lineNum, line in enumerate(code):

            line = re.sub('\t', '', line)

            splitted = re.findall(split_compiler, line)

            if not self._is_assignment(splitted):

                continue
            assign = splitted.index('=')
            var = ''.join(splitted[:assign])
            things = splitted[assign+1:]  # things on rhs of tre assignment
            if not keep_original:
                result[lineNum] = (var, things)
            else:
                result[lineNum] = (var, things, line)

        self.content._reset()

        return result

class CodeAnalyzer(Analyzer):

    def __init__(self, content):

        assert isinstance(content,Code) 
        super(CodeAnalyzer, self).__init__(content)

    def _class_analysis(self, tracker, nodes):

        
        params = self.content._get_params()

        if params !='':

            for param in params.split(' '):
                
                nodes.update({param: Node(param, parent=nodes['root'])})

        for var, things, line in self.get_assignment_lines().values():
                for thing in things:
                    if thing in nodes.keys() and var != thing:
                        nodes.update(
                        {var: Node(var, parent=nodes[thing])})

        return nodes

    def _code_analysis(self, tracker, nodes):

        path_compiler = get_compiler('path')

        for var, things, line in self.get_assignment_lines().values():

            has_data = len(re.findall(path_compiler,line))
            if has_data:
                tracker[var].append(line)
                nodes.update({var: Node(var, parent=nodes['root'])})

            for thing in things:
                if thing in nodes.keys() and var != thing:
                    nodes.update(
                        {var: Node(var, parent=nodes[thing])})
                    
        return nodes

    def _analysis(self):
        

        data_tracker = defaultdict(list)
        root = Node('root')
        node_dictionary = {'root': root}
        is_class = isinstance(self.content, ClassCode)

        if is_class:

            return self._class_analysis(data_tracker, node_dictionary)
        else:

            return self._code_analysis(data_tracker, node_dictionary)

    def _variable_analysis(self, with_package=False):
        '''
        track package usage

        '''

        
        var_counter = defaultdict(list)


        for lineNum, t in self.get_assignment_lines().items():

            var_counter[t[0]].append(lineNum)

            for thing in t[1]:

                if thing in var_counter.keys():
                    var_counter[thing].append(-lineNum)

        return var_counter
                    
class PackageAnalyzer(Analyzer):

    def __init__(self, content, package_dict):
        
        super(PackageAnalyzer, self).__init__(content)
        self.package_dict = package_dict

    
    def _package_analysis(self):
        '''
        extracting packages and track package use

        '''
        package_dicts = self.package_dict
        aliases = list(package_dicts.keys())

        alias_var = defaultdict(list)
        var_alias = defaultdict(list)

        func_counter = defaultdict(bool)

        func_compiler = get_compiler('func')

        for var, things, lineNum in self.get_assignment_lines().values():

            line = var+''.join(things)

            has_func = len(re.findall(func_compiler, line))

            func_counter[lineNum] = has_func != 0

            # if there is a function,only consider the first alias

            related = [thing for thing in things if thing in aliases]

            if has_func:
                
                alias_var[related[0]].append(var)
            
            else:

                for r in related:

                    alias_var[r].append(var)
                    var_alias[var].append(r)

        return alias_var, var_alias

class MarkdownAnalyzer(Analyzer):

    def __init__(self, content):
        
        super(MarkdownAnalyzer,self).__init__(content)

    def _markdown_analysis(self):
        '''
        Line content always follow the previous header

        '''

        header_level = defaultdict(list)
        header_content = defaultdict(list)
        current_header = ''
        md = [ cell['source'] for cell in self.content]
        for lineNum, line in enumerate(md):

            if line.startswith('#'):

                header_level[line.count('#')].append(lineNum)
                current_header = line

            else:

                line = re.sub(get_compiler('image'), re.sub(
                    get_compiler('url'), '', line), line)
                header_content[current_header].append(line)

        return header_level, header_content





class NotebookAnalyzer():

    def __init__(self):

        self.analyzers = ''
        self.notebooks = []
        self.results = []

    def load(self, input:Notebook):
        '''
        import must be a single Notebook
        '''

        self.notebooks.append(input)


    def analyze(self,analyzer_name):
        '''
        analyzers:
        - names for analyzers,list-like
        '''



        assert len(self.notebooks)!=0,'There are no notebooks to be analyzed'
        assert analyzer_name in ['code','package','markdown']
        nb = self.notebooks[-1]
        if analyzer_name == 'code':

            # check if there are valid codes and code blocks

            result = []

            if len(nb.code._get_content()):
                
                ca = CodeAnalyzer(nb.code)
                result.append((ca._analysis(),ca._variable_analysis()))

            else:

                result.append(('',''))



            if len(nb.block_code):

                for cc in nb.block_code:

                    ca = CodeAnalyzer(cc)

                    result.append((ca._analysis(),ca._variable_analysis()))
            
            self.results.append(result)
            self.notebooks.pop()

        if analyzer_name == 'package':

            result = []
            pa = PackageAnalyzer(nb.code, nb.package)
            alias_var, var_alias = pa._package_analysis()
            result.append((alias_var,var_alias))

            if len(nb.block_code):

                for cc in nb.block_code:
                    pa = PackageAnalyzer(nb.code, nb.package)
                    alias_var, var_alias = pa._package_analysis()
                    result.append((alias_var, var_alias))

            self.results.append(result)
            self.notebooks.pop()

