import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())

from data_diagnostics import _sense_checks as sc

CHECK_DICT = {
    'check_name': sc.sense_check(
        func=sc.ratio_xy,
        func_kwargs = {'x': 'A', 'y': 'C'},
        bounds= (.3, .6),
        gen_source=None
    ),
    'sum_check': sc.sense_check(
        func=sc.sum_total,
        func_kwargs= {'total_name': 'F', 'component_names': ['A','C']},
        bounds=(-1,1),
        gen_source=None
    )
}