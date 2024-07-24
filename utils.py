import json
import inspect
BASE_DEPTH = len(inspect.stack(0))
import pandas as pd
from io import TextIOWrapper
import numpy as np
import datetime as dt

def logErrors(error_log_file, error_dict, **kwargs):
    '''
    Wrapper function that returns a string error message for error logging

    :return: str, reports found error when executing a function
    '''
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                error_msg = ':'.join((f.__name__,str(e)))
                if type(e).__name__ not in error_dict:
                    error_dict[type(e).__name__] = set()
                if str(e) not in error_dict[type(e).__name__]:
                    log_and_print(error_msg, debug=True, logger=error_log_file)
                error_dict[type(e).__name__].add(str(e))
        return applicator
    return decorate


def string_dict(d):
    if len(d)==0:
        return "{}"

    center = max([len(str(k)) for k in d.keys()]) + 3

    for k, v in d.items():
        left_pad = center - len(str(k))
        line = ' '*left_pad+ '{} : {}'.format(str(k),str(v))+'\n'
        yield line


def log_and_print(msg='', debug=False, debug_level=99, logger=None, len_adj=0, end='\n', **kwargs):
    '''
    prints msg to console and writes msg to a logger file
    '''
    indent = len(inspect.stack(0))-BASE_DEPTH
    
    if type(msg) not in [str, dict, pd.DataFrame]:
        msg = str(msg)

    if type(msg)==str:
        msg = indent*'-' + " " + msg
    elif type(msg)==dict:
        msg = ''.join([indent*'-' +" "+x for x in string_dict(msg)]).strip('\n')
    elif type(msg)==pd.DataFrame:
        msg = '\n'.join([indent*'-'+" "+ l for l in msg.to_string().split('\n')])
        
    if debug and indent<=debug_level:
        print(msg, end=end)

    if not type(logger)==TextIOWrapper:
        return
    logger.write(msg+'\n')

class NpEncoder(json.JSONEncoder):
    """
    Encoder to export objects that include numpy object types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, type(pd.NaT)):
            return None
        elif isinstance(obj, dt.datetime) or isinstance(obj, pd.Timestamp) or isinstance(obj, dt.date):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)
