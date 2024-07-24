import schema
import os
import pandas as pd
#os.chdir('..')


import run_diagnostics
from diagnostic import diagnostics

try:
    d = {'A': 6}
    d['B']
except Exception as e:
    print(type(e).__name__)
    print(str(e))

if False:
    print('test import')
    folder = 'test_data'

    schema_dict = schema.import_schema(os.path.join(os.getcwd(), 'data_diagnostics', 'test_data'), 'test_schema.xlsx')
    test_data = pd.read_excel(os.path.join(os.getcwd(), 'data_diagnostics', 'test_data', 'test_table.xlsx'))
    from test_data.sense_checks import CHECK_DICT

    x = diagnostics(test_data, schema_dict, 'ID', sense_checks=CHECK_DICT)
    x.run_diagnostics(name_columns='ID')

    #print(x.column_dict)
    print(x.comparison_set)
    #run_diagnostics.enforce_schema(test_data, schema_dict)

