import os
import sys
import json
import numpy as np
import pandas as pd
import functools
import re
import gc
from collections import Counter, defaultdict


class NotebookParser():

    def __init__(self, compilers):
        '''
        compilers: dictionary of regex compilers
        '''
        self.compilers = compilers

    def parse_comment(self, text, comment_type='ns'):
        '''
        text:
        - text to be parsed
        - can be instances of iterables of str or one complete string

        comment_type:
        - comment_types to be parsed out, a string
        - available type: 'ns','dq','tq'
        - default: 'ns'

        returns the parsed out comments and the filtered text

        '''

        text = ''.join(text)

        comment_compiler = self.compilers[comment_type]

        comments = re.findall(comment_compiler, text)

        return ''.join(comments), re.sub(comment_compiler, '', text)

    def parse_package(self, text):
        '''
        text:
        - text to be parsed
        - can be instances of iterables of str or one complete string

        returns the alias:package dict

        '''

        imports = []
        alias_package = {}

        if ' as ' in text:

            imports = re.findall(self.compilers['package_alias'], text)

        else:

            imports = re.findall(self.compilers['package'], text)

        if not len(imports):

            return {}

        package_names = re.sub(' ', '', imports[0]).split(',')

        if len(imports) > 1:

            aliases = re.sub(' ', '', imports[1]).split(',')
        else:
            aliases = []

        for i in range(len(package_names)):

            if i > len(aliases)-1:

                alias_package.update({package_names[i]: package_names[i]})

            else:

                alias_package.update({aliases[i]: package_names[i]})

        return alias_package
