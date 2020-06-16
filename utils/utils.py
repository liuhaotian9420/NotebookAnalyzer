import pandas as pd
import numpy as np
import re
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


def to_csv(content):
    cdata = pd.DataFrame.from_dict(content)
    cdata['id'] = cdata.apply(
        lambda x: x['metadata']['id' if 'id ' in x['metadata'].keys() else 'None'])
    return cdata


def get_compiler(compiler_name):

    return compilers[compiler_name]
