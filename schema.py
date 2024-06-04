from utils import logErrors
import os
import pandas as pd
import warnings
from fuzzywuzzy import process
import inspect
import datetime

class criterion():
    def __init__(self, criterion, **kwargs):
        '''
        criterion could be a list, int, string, or callable
        '''
        self.criterion = criterion
    
    def evaluate(self, value):
        ...

    def enforce(self, value):
        ...


class data_type(criterion):
    def __init__(self, criterion, **kwargs):
        class_translate = {'str': str, 
                           'int': int, 
                           'float': float,
                           'bool': bool, 
                           'date': datetime.date,
                           'datetime': datetime.datetime}
        if criterion in class_translate.keys():
            criterion = class_translate[criterion]

        if not inspect.isclass(criterion):
            raise TypeError(f'data_type criterion must be a data type class. {type(criterion)} {criterion} provided')
        super().__init__(criterion)         

    def evaluate(self, value):
        if pd.isnull(self.criterion): return True
        if pd.isnull(value): return True
        return isinstance(value, self.criterion)
            
    def enforce(self, value):
        if pd.isnull(self.criterion): return value
        if self.criterion==int:
            try: return int(value)
            except:
                try: return pd.to_numeric(value)
                except: return None
        elif self.criterion==str:
            if pd.isnull(value): return value
            try: return str(value)
            except: None
        elif self.criterion==bool:
            try: return bool(value)
            except: None
        elif self.criterion==float:
            try: return float(value)
            except: 
                try: return pd.to_numeric(value)
                except: None
        elif self.criterion==datetime.datetime:
            try: return pd.to_datetime(value)
            except: return None
        elif self.criterion==datetime.date:
            try: return pd.to_datetime(value).date()
            except: return None


class nullable(criterion):
    def __init__(self, criterion, **kwargs):
        criterion = bool(criterion)
        if not isinstance(criterion, bool):
            raise TypeError(f'nullable criterion must be a bool or str bool. {type(criterion)} {criterion} provided')
        super().__init__(criterion)

    def evaluate(self, value):
        if pd.isnull(self.criterion): return True
        if self.criterion:
            return True

        return pd.isnull(value)==self.criterion

    def enforce(self, value):
        return value


class function(criterion):
    ...


class min(function):
    def __init__(self, criterion, **kwargs):
        
        if not isinstance(criterion, int) and \
            not isinstance(criterion, datetime.date) and \
            not isinstance(criterion, datetime.datetime) and \
            not isinstance(criterion, float):
            raise TypeError(f'min criterion must be an integer. {type(criterion)} {criterion} provided')
        super().__init__(criterion)

    def evaluate(self, value):
        if pd.isnull(self.criterion): return True
        try:
            return value>=self.criterion
        except:
            return False

    def enforce(self, value):
        if pd.isnull(self.criterion): return value
        if self.evaluate(value):
            return value
        return None


class max(function):
    def __init__(self, criterion, **kwargs):
        if not isinstance(criterion, int) and \
            not isinstance(criterion, datetime.date) and \
            not isinstance(criterion, datetime.datetime) and \
            not isinstance(criterion, float):
            raise TypeError(f'max criterion must be an integer. {type(criterion)} {criterion} provided')
        super().__init__(criterion)

    def evaluate(self, value):
        if pd.isnull(self.criterion): return True
        try:
            return value<=self.criterion
        except:
            return False
        
    def enforce(self, value):
        if pd.isnull(self.criterion): return value
        if self.evaluate(value):
            return value
        return None
        

class max_len(criterion):
    def __init__(self, criterion, **kwargs):
        criterion = int(criterion)
        if not isinstance(criterion, int):
            raise TypeError(f'max_len criterion must be an integer. {type(criterion)} {criterion} provided')
        super().__init__(criterion)

    def evaluate(self, value):
        if pd.isnull(self.criterion): return True
        try:
            return len(value)<=self.criterion
        except: return False
    
    def enforce(self, value):
        if pd.isnull(self.criterion): return value
        if not self.evaluate(value):
            return str(value)[:self.criterion]
        return value


class category(criterion):
    def __init__(self, name, path, **kwargs):
        import json

        if not os.path.join(os.path.join(path, 'category_files', name)):
            raise ValueError(f'Category json not found in category_files. {name} provided')
        
        ft = name.split('.')[1]
        if ft=='json':
            with open(os.path.join(path, 'category_files', name), 'r') as f:
                criterion = json.load(f)
        elif ft=='txt':
            with open(os.path.join(path, 'category_files', name), 'r') as f:
                criterion = [x.strip() for y in f.readlines() for x in y.split(',')]
        else:
            raise TypeError('File type {ft} is not supported currently.')
        super().__init__(criterion)

    def evaluate(self, value):
        if len(self.criterion)==0: return True
        return value in self.criterion
    
    def enforce(self, value):
        if len(self.criterion)==0: return value

        if self.evaluate(value):
            return value
        
        bestGuess = process.extractBests(value, self.criterion, score_cutoff=70, limit=1)
        if len(bestGuess)==0:
            return None
        
        return bestGuess[0][0]
    

CRITERIA_TYPES = {
    'data_type': data_type,
    'nullable': nullable, 
    'min': min, 
    'max': max, 
    'max_len': max_len,
    'category': category, 

}


def import_schema(path, filename='schema.xlsx', column_name='column_name'):
    sheet_schema = pd.read_excel(os.path.join(path, filename))
    sheet_schema = sheet_schema.set_index(column_name).to_dict(orient='index')

    # Convert sheet_schema into a dict
    ## For each row 
    for k, v in sheet_schema.items():
        for crit_type, criterion in v.items():
            if pd.isnull(criterion):
                continue
            if crit_type not in CRITERIA_TYPES:
                warnings.warn(f'imported schema includes not supported criteria_type: {crit_type}')
                continue
            sheet_schema[k][crit_type] = CRITERIA_TYPES[crit_type](criterion, **{'path': path})


    return sheet_schema


def create_schema(data:pd.DataFrame):

    ...

def export_schema():
    ...




