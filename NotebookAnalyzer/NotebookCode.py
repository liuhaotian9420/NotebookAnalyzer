import numpy
from collections import defaultdict


class Code():

    def __init__(self, content):

        self.pointer = 0

        self.content = content.split('\n')

    def __repr__(self):

        return '\n'.join(self.content)

    def __iter__(self):

        return self

    def __next__(self):

        result = self._get_line(self.pointer)

        self.pointer += 1

        return result

    def _get_line(self, i):

        return self.content[i]


class ClassCode(Code):

    '''
    for code that is in class or in def
    '''

    def __init__(self, content,name, params):
        super(ClassCode, self).__init__(name)
        self.content = content.split('\n')
        self.name = name
        self.params = params

    def _get_name(self):
        return self.name

    def _get_params(self):
        return self.params
