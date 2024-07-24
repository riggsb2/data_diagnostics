
import os
import sys
import time
import getpass
import pandas as pd
import json

from diagnostic import diagnostics
import schema

username = getpass.getuser()
sys.path.append(rf'C:\Users\{username}\AP-Networks\Benchmarking Group - Documents\Data Quality\Diagnostics')

# stops addressed warnings from being shown
pd.options.mode.chained_assignment =None

if __name__ == '__main__':
    #try:
    config_folder = r'C:\Users\briggs\AP-Networks\Benchmarking Group - Documents\Data Quality\Diagnostics\project_diagnostic_config'

    import os
    import time

    today = time.strftime('%Y%m%d', time.localtime())
    OUTPUT_FOLDER = '_'.join((config_folder.split('\\')[-1], today))

    x = diagnostics(config_folder_path=config_folder, output_folder=OUTPUT_FOLDER, index_column='Project_Table_ID')
    x.index_column = 'Project_ID'
    x.name_columns = ['Company','Facility','ProjectName']
    x.run_diagnostics()

    x.summary_report()

    x.export()

    #except Exception as e:
     #   print('Main error')
      #  print(e)

    print('Done')
