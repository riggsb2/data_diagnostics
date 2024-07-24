import pandas as pd
import os
import sys
import json
import re
from importlib import import_module
from statsmodels.stats.stattools import medcouple
import numpy as np
import math as m
import dill
import schema
import time
from utils import logErrors, NpEncoder, log_and_print



class diagnostics():
    def __init__(self, config_folder_path=None, output_folder=None, base_data:pd.DataFrame=None, schema_dict:dict=None, 
                 index_column:str=None, sense_check_dic:dict=None, id_columns:list=None, 
                 name=None,  **kwargs):
        
        self.config_folder_path = config_folder_path
        self.output_folder = output_folder
        if config_folder_path is not None:
            self.load_config_folder(config_folder_path)
        else:
            self.name = name
            self.base_data = base_data
            self.schema_dict = schema_dict
            self.id_columns = id_columns
            self.sense_checks = sense_check_dic

        self.index_column = index_column     
        self.base_data = self.base_data.set_index(self.index_column)   

        self.name_columns = None
        self.obs_dict = {} # {obs: {column: [failed_criterions...]}}
        self.column_dict = {} # {column: {criterion: {quality: , len:  }}}

        self.overall_score = 0
        
        self.labelled_badData = []
        self.comparison_set = []

    def load_config_folder(self, config_folder_path, **kwargs):
        if not os.path.isdir(self.config_folder_path):
            raise ValueError(f'config folder not found at {config_folder_path}')
        
        with open(os.path.join(config_folder_path, 'config.json'), 'r') as f:
            config = json.load(f)
            self.base_data = pd.read_excel(config['data_path'])
        
        sys.path.append(config_folder_path)
        from sense_checks import CHECK_DICT
        self.sense_checks = CHECK_DICT
        self.schema_dict = schema.import_schema(config_folder_path, **kwargs)   
            

    
    def run_diagnostics(self, export_path=os.getcwd(), **kwargs):
        if not os.path.isdir(os.path.join(export_path, self.output_folder)):
            os.mkdir(os.path.join(export_path, self.output_folder))
        
        self.export_path = export_path
        LOG_FILE = open(os.path.join(self.export_path, self.output_folder, 'run_log.log'), 'w')
        ERROR_LOG = open(os.path.join(self.export_path, self.output_folder, 'error_log.log'),'w')
        ERROR_DICT = {}

        @logErrors(ERROR_LOG, ERROR_DICT)
        def update_attributes(column:str, criterion_name:str, series:pd.Series, ):
            '''
            Given a column, criterion name, and series of bool, record in the obs_dict and column_dict
            '''
            failed = series[(series==False)&(series.notnull())]

            for idx in failed.index:
                if idx not in self.obs_dict:
                    self.obs_dict[idx] = {}
                if column not in self.obs_dict[idx]:
                    self.obs_dict[idx][column] = []
                self.obs_dict[idx][column].append(criterion_name)

            if column not in self.column_dict:
                self.column_dict[column] = {}
            if criterion_name not in self.column_dict[column]:
                self.column_dict[column][criterion_name]={'Quality': sum(passfail)/len(passfail), 'length': len(passfail)}
            
        @logErrors(ERROR_LOG, ERROR_DICT)
        def duplicate_check(data, column_display_order=None):
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

                keep_list.extend(self.name_columns)
                # drop any rows that have the same values between dup instances
                dup_df['Drop'] = dup_df.apply(lambda x: True if len(x.unique())==1 and x.name not in keep_list else False, axis=1)
                dup_df = dup_df[dup_df['Drop']==False]

                dup_df = dup_df.drop(columns='Drop')
                return dup_df

            log_and_print('finding duplicates', logger=LOG_FILE)
            dup_review = data.copy()
            # remove any observations already marked as a duplicate
            if 'badData' in dup_review.columns:
                dup_review = dup_review[dup_review['badData']!=2]

            if not isinstance(self.name_columns, list):
                self.name_columns = [self.name_columns]

            # find duplicates with respect to dup_name_columns       
            dup_review['check_index'] = dup_review[self.name_columns].apply(lambda x: tuple(x), axis=1)

            # count number of instances a dup_name_column set appears. keep any that appear more than once
            dup_counts = dup_review.groupby('check_index').size()
            duplicates = dup_counts.index[dup_counts>1]

            # review each potential set of duplicates and add to comparison_set
            for descriptor in duplicates:
                index_df = dup_review.loc[dup_review['check_index']==descriptor]

                compare_df = df_compare(index_df)
                if column_display_order is not None:
                    first_index = list(set(column_display_order + self.name_columns))
                else:
                    first_index = self.name_columns
                compare_df = compare_df.reindex(first_index+[x for x in compare_df.index if x not in first_index] )
                self.comparison_set.append(compare_df.to_dict())

            log_and_print(f'{len(self.comparison_set)} potential duplicates', logger=LOG_FILE)
            
        @logErrors(ERROR_LOG, ERROR_DICT)
        def assemble_for_error_log(s, crit_type, criterion, passfail):
            s = s[~passfail]
            s.name= 'data'
            
            s = s.to_frame()
            s['criterion_name'] = crit_type
            s['criterion'] = str(criterion)
            s['column'] = c
            return s
        
        @logErrors(ERROR_LOG, ERROR_DICT)
        def outlier_check(column_data):
            non_nans = [x for x in column_data.tolist() if not np.isnan(x)]
            if len(non_nans)>1:
                process = 'OUTLIER'
                # Checks for outliers based on base_data. Uses medcouple adjusted
                # outlier detection (p25+IQ)
                # https://en.wikipedia.org/wiki/Box_plot#Variations

                p25 = column_data.quantile(q=0.25)
                p75 = column_data.quantile(q=0.75)
                mc = medcouple(non_nans)
                
                if mc >=0:
                    lower = p25-1.5*np.exp(-4*mc)*abs(p75-p25)
                    upper = p75+1.5*np.exp(3*mc)*abs(p75-p25)
                else:
                    lower = p25-1.5*np.exp(-3*mc)*abs(p75-p25)
                    upper = p75+1.5*np.exp(4*mc)*abs(p75-p25)

                return [(column_data<=upper) | (column_data>=lower)]
            
        @logErrors(ERROR_LOG, ERROR_DICT)
        def scale_check(column_data):
            # order of magnitude check. Determines OOM of each value in the field and
            # uses the p80 range of the calculated OOMs to set a bounding range

            oom_data = column_data.apply(lambda x: m.floor(m.log(abs(x), 10)) if not np.isnan(x) and x!=0 else np.nan)
            lower_oom = oom_data.quantile(q=0.1)
            upper_oom = oom_data.quantile(q=0.9)

            return [(oom_data<=upper_oom) | (oom_data>=lower_oom)]

        @logErrors(ERROR_LOG, ERROR_DICT)
        def evaluate_schema(c, cls):
            return self.base_data[c].apply(cls.evaluate)
        
        dup_cols = []
        unnamed = []
        errors = []
        for c in self.base_data.columns:
            base_pair = None
            if re.search('\.\d+$', c):
                dup_cols.append(c)
            if re.search('^Unnamed', c):
                unnamed.append(c)
            
            if re.search('^A[A-Z]+_', c):
                base = re.sub('^A', '', c)
                if 'P'+base in self.base_data.columns: base_pair = (c, 'P'+base)
  
            if c not in self.schema_dict.keys():
                log_and_print(f'column {c} does not have a schema associated with it', logger=LOG_FILE)
                continue

            # For schema checks
            for crit_type, cls in self.schema_dict[c].items():
                if pd.isnull(cls): continue
                passfail = evaluate_schema(c, cls)
                update_attributes(c, crit_type, passfail)
                errors.append(assemble_for_error_log(self.base_data[c], crit_type, cls.criterion, passfail))

                # For outlier checks on numeric data
                if crit_type== 'data_type' and cls in ['float', 'int']:
                    passfail = outlier_check(self.base_data[c])
                    update_attributes(c, 'outlier', passfail)
                    errors.append(assemble_for_error_log(self.base_data[c], crit_type, cls.criterion, passfail))

                    passfail = scale_check(self.base_data[c])
                    update_attributes(c, 'scale', passfail)
                    errors.append(assemble_for_error_log(self.base_data[c], crit_type, cls.criterion, passfail))
            
            if base_pair is not None:
                passfail = (
                    (self.base_data[base_pair[0]].notnull()) & \
                    (self.base_data[base_pair[0]].apply(lambda x: str(x).strip()!=''))
                ) & (
                    (self.base_data[base_pair[0]].isnull()) | \
                    (self.base_data[base_pair[0]]=='')
                )
                update_attributes(c, 'ap_dependence', passfail)
                errors.append(assemble_for_error_log(self.base_data[base_pair[0]],
                                                     'dependence',
                                                     base_pair[1],
                                                     passfail))
                    
        log_and_print(f'Duplicate column names {dup_cols}', logger=LOG_FILE)
        log_and_print(f'Unnamed column names {unnamed}', logger=LOG_FILE)

        @logErrors(ERROR_LOG, ERROR_DICT)
        def evaluate_sense(obj):
            results = obj.evaluate(self.base_data)

            if results is None: 
                return (None, None)

            passfail = results.apply(obj.flag)
            return results, passfail
            
        # For sense checks/relations
        for name, obj in self.sense_checks.items():
            output = evaluate_sense(obj)
            if output is None: continue
            results, passfail = output
            errors.append(assemble_for_error_log(results, 'sense_check', name, passfail))

        compiled_errors = pd.concat(errors, axis=0)
        compiled_errors = compiled_errors.sort_index()
        compiled_errors.to_excel(os.path.join(export_path, self.output_folder, 'data_concerns.xlsx'))

        duplicate_check(self.base_data, **kwargs)

        ERROR_LOG.close()
        with open(os.path.join(export_path, self.output_folder, 'error.json'), 'w') as f:
            json.dump(ERROR_DICT, f, cls=NpEncoder)

        
        LOG_FILE.close()
    
    def enforce_schema(data:pd.DataFrame, schema:dict, export_path=os.getcwd()):
        changes = []
        for c in data.columns:
            if c not in schema.keys():
                #print(f'column {c} does not have a schema associated with it')
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

    
    def summary_report(self):
        """
        Prints summary data for:
            * column errors
            * duplicates
            * error types
            * data density
            * overall score
        """

        LOG_FILE = open(os.path.join(self.export_path, self.output_folder, 'summary.log'), 'w')


        log_and_print('Data Size', debug=True, logger=LOG_FILE)
        log_and_print(f'Rows: {len(self.base_data)}', debug=True, logger=LOG_FILE)
        log_and_print(f'Vars: {len(self.base_data.columns)}', debug=True, logger=LOG_FILE)
        log_and_print(f'Size {self.base_data.size}', debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)

        log_and_print(f'Potential Duplicates {len(self.comparison_set)}', debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)
        log_and_print(f'First 5 duplicates by {','.join(self.name_columns)}', debug=True, logger=LOG_FILE)

        show_list = []
        for i in range(5):
            if i >=len(self.comparison_set): break
            dup_dict = self.comparison_set[i]
            for k, v in dup_dict.items():
                show_list.append(pd.Series(v, name=k))
                
                
        log_and_print(pd.concat(show_list, axis=1).T[self.name_columns], debug=True, logger=LOG_FILE)

        log_and_print('', debug=True, logger=LOG_FILE)

        log_and_print('Error Types', debug=True, logger=LOG_FILE)
        # {obs: {column: [failed_criterions...]}}
        error_dict = {}
        for i, column_dict in self.obs_dict.items():
            for criterion_list in column_dict.values():
                for criterion in criterion_list:
                    if criterion not in error_dict.keys():
                        error_dict[criterion]=0    
                    error_dict[criterion]+=1

        log_and_print(error_dict, debug=True, logger=LOG_FILE)
        err_df = pd.Series(error_dict).to_frame()
        err_df = err_df.sort_values(0, ascending=False)
        err_df['Total Pct']  = err_df[0]/self.base_data.size # under counts
        err_df['Pct of Errors'] = err_df[0]/err_df[0].sum()
        log_and_print(err_df, debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)

        # {column: {criterion: {'quality': ,'len': }}}
        column_report = {}
        for c,  crit_dict in self.column_dict.items():
            if c not in column_report.keys():
                column_report[c] = {'Has_Criteria': None, 'Quality': None}
            
            if len(crit_dict)==0:
                column_report[c]['Has_Criteria'] = False
                column_report[c]['Quality'] = None
                continue

            column_report[c]['Has_Criteria'] = True

            passed = 0
            total = 0
            for criterion, crit_dict in crit_dict.items():
                total += crit_dict['length']
                passed += crit_dict['length']*crit_dict['Quality']
            column_report[c]['Quality'] = passed/total

        # {column_name: {Has_criterion, Quality}}
        column_df = pd.DataFrame.from_dict(column_report, orient='index').sort_values('Quality', ascending=True)
        self.overall_score = column_df['Quality'].mean()
        
        log_and_print(f'Overall Score: {self.overall_score}', debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)

        log_and_print('Columns with no criteria', debug=True, logger=LOG_FILE)
        for c in sorted(column_df.index[~column_df['Has_Criteria']].tolist()):
            log_and_print(f'-{c}', debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)

        log_and_print('Variable Quality Score Summary', debug=True, logger=LOG_FILE)
        def bin_quality_scores(x):
            if pd.isnull(x):return None
            for threshold in [0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999, 1]:
                if x <= threshold:
                    return threshold
        log_and_print(f'{len(self.base_data.columns)} variables', debug=True, logger=LOG_FILE)
        bins = column_df['Quality'].apply(bin_quality_scores)
        


        log_and_print(bins.value_counts(dropna=False).sort_index(ascending=False), debug=True, logger=LOG_FILE)
        log_and_print('', debug=True, logger=LOG_FILE)

        log_and_print('25 Worst Performing columns', debug=True, logger=LOG_FILE)
        qual_scores = column_df['Quality'][(column_df['Has_Criteria'].notnull()) & (column_df['Quality']!=1)].head(25)
        # TDOD: Only show scores <50% of non-1 scores
        log_and_print(qual_scores, debug=True, logger=LOG_FILE)
        
    
    def export(self, export_path=os.getcwd()):
        if not os.path.isdir(os.path.join(export_path, self.output_folder)):
            os.mkdir(os.path.join(export_path, self.output_folder))

        output_path = os.path.join(export_path, self.output_folder)

        with open(os.path.join(output_path, 'config_folder_path.json'), 'w') as f:
            json.dump(self.config_folder_path, f)

        with open(os.path.join(output_path, 'observations.json'), 'w') as f:
            json.dump(self.obs_dict, f)

        with open(os.path.join(output_path, 'columns.json'), 'w') as f:
            json.dump(self.column_dict, f)

        with open(os.path.join(output_path, 'comparison_set.json'), 'w') as f:
            json.dump(self.comparison_set, f, cls=NpEncoder)
        
        print('Duplicate review report')
        with pd.ExcelWriter(os.path.join(output_path, 'Duplicates_to_review.xlsx')) as writer:
            for couple in self.comparison_set:
                tab_name = [str(x) for x in couple.keys()]
                pd.DataFrame(couple).to_excel(writer, sheet_name='_'.join(tab_name))

def import_diagnostics(folder_name):
    with open(os.path.join(folder_name, 'config_folder_path.json'), 'f') as f:
        config_folder_path = json.load(f)

    with open(os.path.join(folder_name, 'observations.json'), 'r') as f:
        obs_dict = json.load(f)

    with open(os.path.join(folder_name, 'columns.json'), 'r') as f:
        column_dict = json.load(f)

    with open(os.path.join(folder_name, 'comparison_set.json'), 'r') as f:
        comparison_set = json.load(f)

    d = diagnostics()
    d.obs_dict = obs_dict
    d.column_dict = column_dict
    d.comparison_set = comparison_set

    return d


