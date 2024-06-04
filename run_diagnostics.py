
import os
import sys
import time
import getpass
import pandas as pd

from diagnostic import diagnostics
import schema

username = getpass.getuser()
sys.path.append(rf'C:\Users\{username}\AP-Networks\Benchmarking Group - Documents\Data Quality\Diagnostics')

# stops addressed warnings from being shown
pd.options.mode.chained_assignment =None

if __name__ == '__main__':
    try:
        today = time.strftime('%Y%m%d', time.localtime())
        config_folder = r'C:\Users\briggs\AP-Networks\Benchmarking Group - Documents\Data Quality\Diagnostics\data_diagnostics\test_data'
        data = pd.read_excel(os.path.join(os.getcwd(), 'data_diagnostics', 'test_data', 'test_table.xlsx'))
        index_columns = None
        
        sys.path.append(config_folder)
        from sense_checks import CHECK_DICT
        schema_dict = schema.import_schema(config_folder)

        x = diagnostics(data, schema_dict, index_columns, sense_checks=CHECK_DICT)
        x.run_diagnostics(name_columns=['ID'])

    except Exception as e:
        print('Main error')
        print(e)

    print('Done')
    input()
