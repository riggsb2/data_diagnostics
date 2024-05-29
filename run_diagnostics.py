
import os
import re
import sys
import time
import json
#import dill
import getpass
import traceback
import math as m
import numpy as np
import pandas as pd
from shutil import copy
from copy import deepcopy
from datetime import datetime
from datetime import datetime as dt
from statsmodels.stats.stattools import medcouple

# stops addressed warnings from being shown
pd.options.mode.chained_assignment =None

username = getpass.getuser()


# TODO: Optimize file structure to be used in other scripts more easily

if __name__ == '__main__':
    try:
        today = time.strftime('%Y%m%d', time.localtime())

        username = getpass.getuser()
        run_which = ''
        while str(run_which).lower() not in ['o', 'overall', 'unit', 'u', 'b', 'both', 'file', 'f']:
            run_which = input('(F)ile, (O)verall, (U)nit, or (B)oth:')

        if str(run_which).lower() in ['f', 'file']:
            file = input('Please enter complete filepath for downstream dataset:')
            TA_DF = pd.read_stata(file, convert_dates=False)
            index = None
            while index not in TA_DF.columns:
                index = input('What is the index variable?')

            diag = Downstream_Diagnostics(index=index, base_data=TA_DF)
            print('Run Diagnostics')
            diag.run_diagnostics(check_list=['ALL'])
            print('Save to folder')
            diag.export_report(os.path.join('02_Diagnostic_Runs', '_'.join((today, 'OVERALL'))))

        if str(run_which).lower() in ['o', 'overall', 'b', 'both']:
            print('Read in latest Development Database')
            TA_DF = pd.read_stata(r"C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Stata Files\Downstream\STATA_TA_Development_Database.dta".format(username),
                                    convert_dates=False)
            diag = Downstream_Diagnostics(index='Survey_ID', base_data=TA_DF)
            print('Run Diagnostics')
            diag.run_diagnostics(check_list=['ALL'])
            diag.summary_report()
            print('Save to folder')
            diag.export_report(os.path.join('02_Diagnostic_Runs', '_'.join((today, 'OVERALL'))))

        if str(run_which).lower() in ['u', 'unit', 'b', 'both']:
            print('Read in latest Unit Database')
            Unit_DF = pd.read_stata(r"C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Stata Files\Downstream\STATA_UNITS_Development_Database.dta".format(username),
                                    convert_dates=False)
            print(np.nan in Unit_DF.index)
            unit = Downstream_Diagnostics(index='Survey_Unit_ID', base_data=Unit_DF)
            print('Run Diagnostics')
            unit.run_diagnostics(check_list='ALL')
            print('Save to folder')
            unit.export_report('_'.join((today, 'UNIT')))
    except Exception as e:
        print('Main error')
        print(e)

    print('Done')
    input()
