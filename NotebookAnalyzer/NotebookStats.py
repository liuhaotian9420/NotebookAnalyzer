import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from anytree import Node, RenderTree, LevelOrderGroupIter


def entity_analysis(node_dictionary):

    node_dictionary = node_dictionary

    root = node_dictionary['root']

    orders = [[node.name for node in children]
              for children in LevelOrderGroupIter(root)]

    order_consumption = []

    if len(orders):

        order_consumption = [len(o) for o in orders]

    average_reference_length = 0

    if len(orders) >= 2:
        average_reference_length = np.mean(np.array(
            [grandchildren.height for child in root.children for grandchildren in child.children]))

    max_expansion = 0

    if len(orders)-1:

        prev, cur, level = 1, 1, 1
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

    return {'order_consumption': order_consumption, 'average_reference_length': average_reference_length, 'max_expansion': max_expansion}


def full_analysis(alias_var, var_alias, var_counter, package_dict):
    '''

    Analysis of package usage


    '''

    alias_var, var_alias, var_counter = alias_var, var_alias, var_counter

    alias_package_dict = package_dict

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
