import pandas as pd
import os
from copy import deepcopy

def outlier_check():
    column_data = self.base_data[col]
    non_nans = [x for x in column_data.tolist() if not np.isnan(x)]
    if len(non_nans)>1:
        process = 'OUTLIER'
        # Checks for outliers based on base_data. Uses medcouple adjusted
        # outlier detection (p25+IQ)
        # https://en.wikipedia.org/wiki/Box_plot#Variations
        if process in check_list or 'ALL' in check_list:
            p25 = column_data.quantile(q=0.25)
            p75 = column_data.quantile(q=0.75)
            mc = medcouple(non_nans)
            
            if mc >=0:
                lower = p25-1.5*np.exp(-4*mc)*abs(p75-p25)
                upper = p75+1.5*np.exp(3*mc)*abs(p75-p25)
            else:
                lower = p25-1.5*np.exp(-3*mc)*abs(p75-p25)
                upper = p75+1.5*np.exp(4*mc)*abs(p75-p25)

            self.column_dict[col]['Check_Count']+=1
            outlier_index = column_data.index[(column_data>upper) | (column_data<lower)]

            update_columns(outlier_index, col, 'OUTLIER', 'ids')

            self.column_dict[col]['UPPER_BND'] = upper
            self.column_dict[col]['LOWER_BND'] = lower
            update_obs(outlier_index, col, 'OUTLIER', (lower, upper))

            self.column_dict[col]['Quality'] += obs-len(outlier_index)

def scale_check():
    process = 'SCALE'
    # order of magnitude check. Determines OOM of each value in the field and
    # uses the p80 range of the calculated OOMs to set a bounding range
    if process in check_list or 'ALL' in check_list:

        self.column_dict[col]['Check_Count']+=1
        oom_data = column_data.apply(lambda x: m.floor(m.log(abs(x), 10)) if not np.isnan(x) and x!=0 else np.nan)
        lower_oom = oom_data.quantile(q=0.1)
        upper_oom = oom_data.quantile(q=0.9)

        oom_index = oom_data.index[(oom_data>upper_oom) | (oom_data<lower_oom)]
        update_columns(oom_index, col, 'SCALE', 'ids')
        self.column_dict[col]['UPPER_SCALE'] = upper_oom
        self.column_dict[col]['LOWER_SCALE'] = lower_oom
        update_obs(oom_index, col, 'SCALE', (lower_oom, upper_oom))
        self.column_dict[col]['Quality'] += obs-len(oom_index)

def sense_checks():
    ...

def sum_check():
    ...

class diagnostics():
    def __init__(self, base_data:pd.DataFrame, schema:dict, index_column:str, sense_checks:dict=None,
                 id_columns:list=None):

        self.base_data = base_data
        self.schema = schema
        self.index_column = index_column
        self.id_columns = id_columns
        self.sense_checks = sense_checks

        self.obs_dict = {} # {obs: {column: {criterion: pass/fail}}}
        self.column_dict = {} # {column: {criterion: {obs: pass/fail}}}

        self.overall_score = 0
        
        self.labelled_badData = []
        self.table_dict = {}
        self.comparison_set = []

        self.kwarg_config = {}

    def run_diagnostics(self, export_path=os.getcwd(), **kwargs):
        
        def update_attributes(column:str, criterion_name:str, series:pd.Series, ):
            '''
            Given a column, criterion name, and series of bool, record in the obs_dict
            '''
            for idx, pass_fail in series.to_dict().items():
                if idx not in self.obs_dict:
                    self.obs_dict[idx] = {}
                if c not in self.obs_dict[idx]:
                    self.obs_dict[idx][column] = {}
                self.obs_dict[idx][column][criterion_name] = pass_fail

            if column not in self.column_dict:
                self.column_dict[column] = {}
            if criterion_name not in self.column_dict[column]:
                self.column_dict[column][criterion_name]=series.to_dict()
    

        def duplicate_check(data, name_columns, column_display_order=None):
            # Check for duplicates in data based on Com, Group, fac, ta_name, and session
            def df_compare(dup_df:pd.DataFrame, keep_list:list=[])->pd.DataFrame:
                """
                Compares a dataframe of two suspected duplicate obseratvations

                variable: dup_df: dataframe of duplicate data to check

                returns: dup_df: places where data is different formatted
                """

                def type_missing_check(val:object)->bool:
                    """
                    checks if a value is considered "missing". Incorporates Stata missing str format
                    """
                    if isinstance(val, str):
                        if val=="" or val==".": return True
                        else: return False
                    elif pd.isnull(val): return True
                    else: return False

                dup_df = dup_df.copy()
                # creates a missing variable which counts the number of missing values in an observation
                dup_df.loc[:,'missing'] = dup_df.apply(lambda x: len(dup_df.columns)- x.count(), axis=1)
                dup_df = dup_df.T

                keep_list.extend(name_columns)
                # drop any rows that have the same values between dup instances
                dup_df['Drop'] = dup_df.apply(lambda x: True if len(x.unique())==1 and x.name not in keep_list else False, axis=1)
                dup_df = dup_df[dup_df['Drop']==False]

                dup_df = dup_df.drop(columns='Drop')
                return dup_df

            print('finding duplicates')
            dup_review = data.copy()
            # remove any observations already marked as a duplicate
            if 'badData' in dup_review.columns:
                dup_review = dup_review[dup_review['badData']!=2]

            if not isinstance(name_columns, list):
                name_columns = [name_columns]

            # find duplicates with respect to dup_name_columns       
            dup_review['check_index'] = dup_review[name_columns].apply(lambda x: tuple(x), axis=1)

            # count number of instances a dup_name_column set appears. keep any that appear more than once
            dup_counts = dup_review.groupby('check_index').size()
            duplicates = dup_counts.index[dup_counts>1]

            # review each potential set of duplicates and add to comparison_set
            for descriptor in duplicates:
                index_df = dup_review.loc[dup_review['check_index']==descriptor]

                compare_df = df_compare(index_df)
                if column_display_order is not None:
                    first_index = list(set(column_display_order + name_columns))
                else:
                    first_index = name_columns
                compare_df = compare_df.reindex(first_index+[x for x in compare_df.index if x not in first_index] )
                self.comparison_set.append(compare_df.to_dict())
            
        errors = []
        for c in self.base_data.columns:
            if c not in self.schema.keys():
                print(f'column {c} does not have a schema associated with it')
                continue

            # For schema checks
            for crit_type, cls in self.schema[c].items():
                if pd.isnull(cls): continue
                passfail = self.base_data[c].apply(cls.evaluate)
                update_attributes(c, crit_type, passfail)

                s = self.base_data[c][~passfail]
                s.name= 'data'
                
                s = s.to_frame()
                s['criterion_name'] = crit_type
                s['criterion'] = str(cls.criterion)
                s['column'] = c
                errors.append(s)

            # For sense checks/relations


        compiled_errors = pd.concat(errors, axis=0)
        compiled_errors = compiled_errors.sort_index()
        #compiled_errors.to_excel(os.path.join(export_path, 'data_concerns.xlsx'))

        duplicate_check(self.base_data, **kwargs)

    
    def enforce_schema(data:pd.DataFrame, schema:dict, export_path=os.getcwd()):
        changes = []
        for c in data.columns:
            if c not in schema.keys():
                print(f'column {c} does not have a schema associated with it')
                continue
            for crit_type, cls in schema[c].items():
                if pd.isnull(cls): continue
                old = data[c]
                old.name = 'original'
                new = data[c].apply(cls.enforce)
                new.name= 'changes'
                review = pd.concat([old, new], axis=1)
                review['criterion_name'] = crit_type
                review['criterion'] = str(cls.criterion)
                review['column'] = c
                review = review[(review['original']!=review['changes']) &\
                                (review['original'].isnull()!=review['changes'].isnull())]
                data[c] = new
                changes.append(review)
        compiled_changes = pd.concat(changes, axis=0)
        compiled_changes = compiled_changes.sort_index()
        #compiled_changes.to_excel(os.path.join(export_path, 'data_changes.xlsx'))

