from utils import logErrors
import json
import os

class Downstream_Diagnostics():
    def __init__(self, index, base_data=None, import_path=None):
        """
        If a folder is  provided on init, load existing diagnostics else,
        # begin a new instance of the class

        variables
        ----------
        index: string, variable name to use as the indexer in the base_data
        base_data: pandas.DataFrame, data to be analyzed
        folder: save folder for export of diagnostics

        """
        if import_path and not base_data:
            @logErrors()
            def load_json(json_name):
                with open(os.path.join(import_path, json_name), 'r') as f:
                    return(json.load(f))

            self.index=index
            self.base_data = pd.read_json(load_json('Base_data.json'), orient='split')
            self.column_dict = load_json('variable_diagnostics.json')
            self.obs_dict = load_json('observation_diagnostics.json')
            self.TAN_only_list = load_json('TAN_value_options.json')
            self.criteria_dict = load_json('Diagnostic_criteria.json')
            self.labelled_badData = load_json('labelled_badData.json')
            self.table_dict = load_json('table_diagnostics.json')
            self.comparison_set = load_json('duplicate_check.json')

            self.overall_score = pd.DataFrame.from_dict(self.column_dict, orient='index')['Quality'].mean()
            print('Imported Diagnostics')

        else:
            self.index = index
            self.base_data = base_data
            self.obs_dict = {}
            self.column_dict = {}
            self.overall_score = 0
            self.labelled_badData = []
            self.table_dict = {}
            self.comparison_set = []

            with open(r'C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Data Cleaning\Diagnostics\01_TAN Reference Files\TAN_value_options.json'.format(username), 'r') as f:
                self.TAN_only_list = json.load(f)

            with open(r'C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Data Cleaning\Diagnostics\01_TAN Reference Files\Diagnostic_criteria.json'.format(username), 'r') as f:
                self.criteria_dict = json.load(f)

            print('Import Industry Metrics')
            resource_home = r"C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Data Cleaning\Diagnostics\01_TAN Reference Files".format(username)
            metric_home = os.path.join(resource_home, 'Industry_Metrics')

            self.IND_METRICS ={}
            for fname in [x for x in os.listdir(metric_home) if str(x).endswith('.json')]:
                with open(os.path.join(metric_home, fname), 'r') as f:
                    print(fname)
                    self.IND_METRICS = {**self.IND_METRICS, **json.load(f)}


            print('Import sense checks')
            sense_home = os.path.join(resource_home, 'Sense_Checks')
            with open(os.path.join(sense_home, 'sense_check.p'), 'rb') as f:
                self.sense_checks = dill.load(f)

            max_idx = self.base_data[self.index].max()
            add_idx = self.base_data.index[self.base_data[self.index].isna()]
            if len(add_idx)>0:
                print('Adding {} temporary {} incrementing from the max of {}'.format(len(add_idx), self.index, max_idx))
                add_idx = {k: max_idx+i for i, k in enumerate(add_idx)}
                self.base_data.fillna(add_idx, inplace=True)
                
            print('Set up Diagnostics')


    def run_diagnostics(self, check_list=[]):
        """
        Runs diagnostics on the dataset including:
            * TAN_name alignment
            * If obs can be zero or null
            * If the sum of component variables are greater than the total
            * If order of magnitude is reasonable
            * If the field is the right data type
            * If the field is outside an Industry Metrics
            * If the field is outside a sense check
            * if the field is an outlier given the base_data
            * Records existing labels of badData
            * marks duplicate entries
            * calculates quality score of each field

        check_list: list, list of processes to include. 'ALL' will run all checks

        """

        def update_obs(indices, col, check, criteria):
            """
            Update the observation diagnostic table with an evaluation

            variable
            ---------
            indices: list, indices of the observations that are out of bounds
            col: string, the variable that has out of bounds data
            check: string, type of check being conducted
            criteria: dictionary, criteria dictionary to cite in variable entry
            """
            if re.search('ID', col):
                print('ID variable, skipping', col)
                return

            if col not in self.column_dict.keys():
                new_column(col)

            for i in indices:
                if i not in self.obs_dict.keys():
                    self.obs_dict[i] = {}
                    if 'Survey_BMID' in self.base_data.columns:
                        self.obs_dict[i]['Survey_BMID'] = self.base_data.loc[i, 'Survey_BMID']
                    self.obs_dict[i][self.index] = self.base_data.loc[i, self.index]

                if col not in self.obs_dict[i].keys():
                    self.obs_dict[i][col]={'value': self.base_data.loc[i, col]}

                # handles the form of the criteria passed to the function
                if isinstance(criteria, pd.core.series.Series):
                    add_dict = {check: criteria[i]}
                elif isinstance(criteria, pd.core.frame.DataFrame):
                    add_dict = {check: criteria.loc[i,:].to_dict()}
                elif check in self.obs_dict[i][col].keys():
                    if type(self.obs_dict[i][col][check])==list:
                        add_dict = {check: self.obs_dict[i][col][check] +[criteria]}
                    else:
                        add_dict = {check: [self.obs_dict[i][col][check]] + [criteria]}
                else:
                    add_dict = {check: criteria}


                self.obs_dict[i][col] = {**self.obs_dict[i][col], **add_dict}
                del add_dict


        def new_column(col):
            """
            Creates new column for the column diagnostics
            """
            self.column_dict[col] = {'Has_Criteria': None, 'Quality': 0, 'Check_Count': 0}


        def update_columns(indices, col, check, dict_type='values'):
            """
            Update the observation diagnostic table with an evaluation

            variable
            ---------
            indices: list, indices of the observations that are out of bounds
            col: string, the variable that has out of bounds data
            check: string, type of check being conducted
            dict_type: string, type of check being conducted, affects how error is reported
            """

            if col not in self.column_dict.keys():
                new_column(col)

            if dict_type=='values':
                summary = self.base_data[col][indices].value_counts(dropna=False).to_dict()
            elif dict_type=='ids':
                if 'Survey_BMID' in self.base_data.columns:
                    summary = self.base_data.loc[indices, [col, 'Survey_BMID']].to_dict(orient='index')
                else:
                    summary = self.base_data.loc[indices, [col, self.index]].to_dict(orient='index')

            self.column_dict[col][check]=summary


        try:
            print('-', time.strftime('%H:%M:%S',time.localtime()), 'Start Diagnostics')
            cnt = 0
            total = len(self.base_data.columns)
            obs = len(self.base_data)
            # review each variable in the data set and review for errors
            for col in self.base_data.columns:
                score = 0
                cnt+=1
                bounds=False
                process='START'
                if col not in self.column_dict.keys():
                        new_column(col)
                if col in self.criteria_dict.keys():

                    self.column_dict[col]['Has_Criteria']= True
                    IS_NULLABLE = self.column_dict[col]['Has_Criteria']
                    
                    empty_count = self.base_data[col].isna().sum()

                    process='TABLE'
                    # Missing data table diagnostics
                    table_name = self.criteria_dict[col]['TABLE_NAME']
                    if table_name:
                        # If the column is actual data. Slim what's applicable
                        # for diagnostics
                        if re.search('^A', col):
                            applicable_data = self.base_data[col][self.base_data['DC_Session']!='Look Back']
                        else:
                            applicable_data = self.base_data[col]

                        tuple_to_add = (applicable_data.isna().sum(), applicable_data.size)
                        if table_name in self.table_dict.keys():
                            self.table_dict[table_name].append(tuple_to_add)
                        else:
                            self.table_dict[table_name] = [tuple_to_add]

                    process = 'TAN_ACCEPTED'
                    # TAN alignment check: For values that come from a dropdown list, check if in TAN accepted values
                    if process in check_list or 'ALL' in check_list:
                        self.column_dict[col]['Check_Count']+=1
                        if self.criteria_dict[col]['COLUMN_NAME'] in self.TAN_only_list.keys():
                            TAN_accepted = self.TAN_only_list[self.criteria_dict[col]['COLUMN_NAME']]
                            # Add 'missing' to accepted values if field can be missing
                            if IS_NULLABLE:
                                TAN_accepted.append('')
                                TAN_accepted.append('NaN')
                                TAN_accepted.append(np.nan)

                            align_index = self.base_data.index[~self.base_data[col].isin(TAN_accepted)]
                            update_columns(align_index, col, 'TAN_ACCEPTED')
                            update_obs(align_index, col, 'TAN_ACCEPTED', 'See JSON')
                            self.column_dict[col]['Quality'] += obs-len(align_index)
                        else:
                            self.column_dict[col]['Quality'] += obs

                    process = 'NULLABLE'
                    # Null check: If column can't be null, return index where null
                    if process in check_list or 'ALL' in check_list:
                        IS_NULLABLE = self.criteria_dict[col]['IS_NULLABLE']
                        self.column_dict[col]['Check_Count']+=1
                        if not IS_NULLABLE:
                            null_index = self.base_data.index[self.base_data[col].isna()].tolist()
                            update_columns(null_index, col, 'IS_NULLABLE')
                            update_obs(null_index, col, 'IS_NULLABLE', IS_NULLABLE)
                            self.column_dict[col]['Quality'] += obs-len(null_index)
                        else:
                            self.column_dict[col]['Quality'] += obs

                    process='ZEROABLE'
                    # Zero check: If column can't be zero, return index where 0
                    if process in check_list or 'ALL' in check_list:
                        IS_ZEROABLE = self.criteria_dict[col]['IS_ZEROABLE']
                        self.column_dict[col]['Check_Count']+=1
                        if not IS_ZEROABLE:
                            zero_index = self.base_data.index[self.base_data[col]==0].tolist()
                            empty_count += len(zero_index)
                            update_columns(zero_index, col, 'IS_ZEROABLE')
                            update_obs(zero_index, col, 'IS_ZEROABLE', IS_ZEROABLE)
                            self.column_dict[col]['Quality'] += obs-len(zero_index)
                        else:
                            self.column_dict[col]['Quality'] += obs

                    process = 'DATATYPE'
                    # Data Type check: If column doesn't match proper datatype, return index where mismatch
                    if process in check_list or 'ALL' in check_list:
                        DATA_TYPE = self.criteria_dict[col]['DATA_TYPE']
                        self.column_dict[col]['Check_Count']+=1
                        type_test=None
                        column_data = self.base_data[col].where(self.base_data[col].notnull(), False)

                        if DATA_TYPE in ['nvarchar', 'nchar']:
                            type_test = column_data.apply(lambda x: isinstance(x, str) | (not x))
                        elif DATA_TYPE in ['float', 'decimal']:
                            type_test = column_data.apply(lambda x: isinstance(x, float) | (not x))
                            bounds=True
                        elif DATA_TYPE in ['smalldatetime', 'datetime']:
                            type_test = column_data.apply(lambda x: (isinstance(x, (pd._libs.tslibs.nattype.NaTType, datetime, float))) | (not x))
                        elif DATA_TYPE in ['bit', 'int']:
                            type_test = column_data.apply(lambda x: True if not x else (int(x)==x if type(x) in [int, float] else isinstance(x, int)))
                        else:
                            print('No type listed for', col)

                        type_index = type_test.index[type_test==False]
                        update_columns(type_index, col, 'DATA_TYPE', 'ids')
                        update_obs(type_index, col, 'DATA_TYPE', DATA_TYPE)
                        self.column_dict[col]['Quality'] += obs-len(type_index)

                # Quantitative bounds like scaling and outliers for numeric col types
                if self.base_data[col].dtype in [float, int]:
                    try:
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

                    except Exception as e:
                        print(col)
                        print(e)

                # status reporter
                if cnt%100==0:
                    print('-', time.strftime('%H:%M:%S',time.localtime()), str(int(100*cnt/total)))

                # runaway escape
                if cnt==9999:
                    break

            process='EMAIL'
            # Check formatting of emails based on a regex
            if process in check_list or 'ALL' in check_list:
                email_reg = r'([A-Za-z0-9\-\_]+\.?)+\@([A-Za-z0-9\-\_]+\.?)+\w{2,3}'
                email_index = self.base_data.index[~self.base_data['Email'].astype(str).apply(lambda x: True if re.search(email_reg, x) or x=='' else False)]
                email_summary = self.base_data['Email'][email_index].value_counts().to_dict()
                self.column_dict['Email']['FORMATTING']=email_summary
                self.column_dict['Email']['Quality'] += obs - len(email_index)

            process='SUMS'
            # Check if component sums are greater than or equal to their total variable
            if process in check_list or 'ALL' in check_list:
                print('Check of sum ')
                sum_check_dict = {
                             'C_Main_TOTAL': ['C_Main_Directs', 'C_Main_Equip_Rental',  'C_Main_Indirects',
                                          'C_Main_Logistics', 'C_Main_Materials', 'C_Main_Planning_Prep',
                                          'C_Main_PreTA_Exe'],
                             'C_Cap_During_TOTAL': ['C_Cap_Directs', 'C_Cap_Indirects',  'C_Cap_Materials'],
                             'C_TA_TOTAL': ['C_Main_TOTAL', 'C_Cap_During_TOTAL'],
                             'C_Main_DCL_TOTAL': ['C_Main_DCL_Boilermaker','C_Main_DCL_Bricklayer_Refract','C_Main_DCL_Catalyst_Install',
                                                  'C_Main_DCL_Civil','C_Main_DCL_Elect_Instrum','C_Main_DCL_Field_Machining',
                                                  'C_Main_DCL_General_Piping','C_Main_DCL_Industrial_Cleaning','C_Main_DCL_Millwrights',
                                                  'C_Main_DCL_Rotating_Eqp','C_Main_DCL_Safety_Watch','C_Main_DCL_Scaffolding',
                                                  'C_Main_DCL_Specialty_Contractor', 'C_Main_DCL_Welders',
                                                  'C_Main_DCL_Other_1','C_Main_DCL_Other_2','C_Main_DCL_Other_3','C_Main_DCL_Other_4',
                                                  'C_Main_DCL_Other_5','C_Main_DCL_Other_6','C_Main_DCL_Other_7'],
                             'C_Main_DSL_TOTAL': ['C_Main_DSL_Craft_Foreman', 'C_Main_DSL_Crane_Op', 'C_Main_DSL_Decontamination',
                                                  'C_Main_DSL_General_Labor', 'C_Main_DSL_Light_Temp_Power', 'C_Main_DSL_NDE_NDT_Crew',
                                                  'C_Main_DSL_Operations_Support', 'C_Main_DSL_Tool_Room_Attend',
                                                  'C_Main_DSL_Other_1', 'C_Main_DSL_Other_2', 'C_Main_DSL_Other_3', 'C_Main_DSL_Other_4',
                                                  'C_Main_DSL_Other_5'],
                             'H_Main_DCL_TOTAL': ['H_Main_DCL_Boilermaker','H_Main_DCL_Bricklayer_Refract','H_Main_DCL_Catalyst_Install',
                                                  'H_Main_DCL_Civil','H_Main_DCL_Elect_Instrum','H_Main_DCL_Field_Machining',
                                                  'H_Main_DCL_General_Piping','H_Main_DCL_Industrial_Cleaning','H_Main_DCL_Millwrights',
                                                  'H_Main_DCL_Rotating_Eqp','H_Main_DCL_Safety_Watch','H_Main_DCL_Scaffolding',
                                                  'H_Main_DCL_Specialty_Contractor','H_Main_DCL_Welders',
                                                  'H_Main_DCL_Other_1','H_Main_DCL_Other_2','H_Main_DCL_Other_3','H_Main_DCL_Other_4',
                                                  'H_Main_DCL_Other_5','H_Main_DCL_Other_6','H_Main_DCL_Other_7'],
                             'H_Main_DSL_TOTAL': ['H_Main_DSL_Craft_Foreman', 'H_Main_DSL_Crane_Op', 'H_Main_DSL_Decontamination',
                                                  'H_Main_DSL_General_Labor', 'H_Main_DSL_Light_Temp_Power', 'H_Main_DSL_NDE_NDT_Crew',
                                                  'H_Main_DSL_Operations_Support', 'H_Main_DSL_Tool_Room_Attend',
                                                  'H_Main_DSL_Other_1', 'H_Main_DSL_Other_2', 'H_Main_DSL_Other_3', 'H_Main_DSL_Other_4',
                                                  'H_Main_DSL_Other_5'],
                             'C_Main_Indirects': ['C_Main_IND_Admin', 'C_Main_IND_Engineering', 'C_Main_IND_HSE',
                                                  'C_Main_IND_Inspectors', 'C_Main_IND_Logistics_Labor', 'C_Main_IND_Materials_Mgt',
                                                  'C_Main_IND_Operations_Rep', 'C_Main_IND_Planner_Estimator', 'C_Main_IND_Schedulers',
                                                  'C_Main_IND_Security', 'C_Main_IND_TA_Mgmt', 'C_Main_IND_Other_1',
                                                  'C_Main_IND_Other_2', 'C_Main_IND_Other_3', 'C_Main_IND_Other_4'],
                             'C_Main_Materials':['C_Main_MAT_Admin_Supplies', 'C_Main_MAT_Bulk', 'C_Main_MAT_Equip_Purchase',
                                                 'C_Main_MAT_Freight_Expediting', 'C_Main_MAT_Gasses', 'C_Main_MAT_Insulation',
                                                 'C_Main_MAT_Ops_Supplies', 'C_Main_MAT_Spare_Parts',
                                                 'C_Main_MAT_Other_1', 'C_Main_MAT_Other_2', 'C_Main_MAT_Other_3',
                                                 'C_Main_MAT_Other_4', 'C_Main_MAT_Other_5'],
                             'H_Main_TOTAL': ['H_Main_DFL', 'H_Main_Indirects'],
                             'H_Cap_Execution_TOTAL': ['H_Cap_DFL_Execution', 'H_Cap_Indirects_Execution'],
                             'H_TA_TOTAL': ['H_Main_TOTAL',  'H_Cap_Execution_TOTAL'],

                            }
                for t in ['P','A']:
                    for total, components in sum_check_dict.items():
                        col = ''.join((t, total))
                        components = [''.join((t, c)) for c in components]
                        components = [x for x in components if x in self.base_data.columns]
                        if col in self.base_data.columns:
                            cumulative_data = self.base_data[components].sum(axis=1)
                            cumulative_index = self.base_data.index[self.base_data[col]<cumulative_data]
                            update_columns(cumulative_index, col, 'COMPONENT_SUM', 'ids')
                            update_obs(cumulative_index, col, 'COMPONENT_SUM', cumulative_data)
                            self.column_dict[col]['Check_Count']+=1
                            self.column_dict[col]['Quality'] += obs-len(cumulative_index)

            process = 'labelled_badData'
            # records any previously marked badData observations
            if process in check_list or 'ALL' in check_list:
                print('IDing previously marked badData')
                if 'badData' in self.base_data.columns:
                    self.labelled_badData = self.base_data['badData'][self.base_data['badData'].notnull()].to_dict()

            process = 'duplicate_check'
            # Check for duplicates in data based on Com, Group, fac, ta_name, and session
            if process in check_list or 'ALL' in check_list:
                def df_compare(duplicate_df):
                    """
                    Compares a dataframe of two suspected duplicate obseratvations

                    variable: duplicates_df: pandas.DataFrame, dataframe of duplicate data to check
                        formatted (obs, variables)

                    returns: tdf: pandas.DataFrame, places where data is different formatted
                        (variables, obs)
                    """

                    def type_missing_check(v):
                        """
                        checks if a value is considered "missing". Incorporates Stata missing str format

                        variable: v: object, value to be checked

                        returns: 1|0
                        """
                        if type(v)==str:
                            if v=="" or v==".":
                                return 1
                            else:
                                return 0
                        elif type(v)==int or type(v)==float:
                            if np.isnan(v) or v==0:
                                return 1
                            else:
                                return 0
                        else:
                            return 0

                    # creates a missing variable which counts the number of missing values in an observation
                    duplicate_df.loc[:,'missing'] = deepcopy(duplicate_df.apply(lambda x: len(duplicate_df.columns)- x.count(), axis=1))

                    tdf = duplicate_df.T
                    keep_list = ['Source', 'Survey_ID', 'Survey_Unit_ID', 'Survey_ID', 'Company', 'Facility',
                                 'TA_Name', 'DC_Session', 'TA_Year', 'Data_Collection_Name']
                    if 'Survey_BMID' in tdf.columns:
                        keep_list.append('Survey_BMID')
                    # drop any rows that have the same values between dup instances
                    tdf['Drop'] = tdf.apply(lambda x: True if len(x.unique())==1 and x.name not in keep_list else False, axis=1)
                    tdf = tdf[tdf['Drop']==False]
                    tdf = tdf.drop(columns='Drop')
                    return tdf


                print('finding duplicates')
                dup_review = deepcopy(self.base_data)
                # remove any observations already marked as a duplicate
                if 'badData' in dup_review.columns:
                    dup_review = dup_review[dup_review['badData']!=2]

                # find duplicates with respect to dup_name_columns
                dup_name_columns = ['Company', 'Facility', 'TA_Name', 'DC_Session']
                if self.index=='Survey_Unit_ID':
                    dup_name_columns.append('Unit_Name_APN_Std')
                    dup_name_columns.append('Unit_Name_Client')
                dup_review['check_index'] = dup_review[dup_name_columns].apply(lambda x: tuple(x), axis=1)

                # count number of instances a dup_name_column set appears. keep any that appear more than once
                survey_counts = dup_review.groupby(dup_name_columns).size()
                duplicates = survey_counts.index[survey_counts>1]

                # review each potential set of duplicates and add to comparison_set
                for descriptor in duplicates:
                    index_df = dup_review[dup_review['check_index']==descriptor]

                    compare_df = df_compare(index_df)
                    first_index = ['Survey_ID', 'Survey_Unit_ID', 'ID', 'Survey_BMID', 'Source'] + dup_name_columns + ['TA_Year', 'Data_Collection_Name']
                    compare_df = compare_df.reindex(first_index+[x for x in compare_df.index if x not in first_index] )
                    self.comparison_set.append(compare_df.to_dict())

            process = "IND_Metrics"
            # checks if IND_Metric variables are outside 2SD or IQR of TINC metrics
            if process in check_list or 'ALL' in check_list:
                def check_ind_metrics(metric_name, metric_data):
                    def assign_metric(x, data={}):
                        """
                        establishes which industry metric values to use
                        """
                        try:
                            ranges = data[','.join((x['TA_Region'], x['TA_Complexity']))]
                        except:
                            try:
                                ranges = data[','.join(('Overall', 'Overall'))]
                            except:
                                return pd.Series({'lo':None, 'avg': None, 'hi':None})

                        if not np.isnan(ranges['p25']) and not np.isnan(ranges['p75']):
                            return pd.Series({'lo': ranges['p25'], 'avg': ranges['avg'], 'hi': ranges['p75']})
                        elif not np.isnan(ranges['-1 stdev']) and not np.isnan(ranges['+1 stdev']):
                            std = ranges['avg'] - ranges['-1 stdev']
                            return pd.Series({'lo': ranges['avg']-2*std, 'avg': ranges['avg'], 'hi': ranges['avg']+2*std})
                        else:
                            return pd.Series({'lo':None, 'avg': None, 'hi':None})


                    def flag(obs, col):
                        """
                        flags observation if value is outside TINC bounds
                        """
                        rdf = {'avg': obs['avg'], 'limit':None, 'flag': False}
                        if obs['hi'] and obs['lo']:
                            if obs[col]<obs['lo']:
                                rdf['limit'] = obs['lo']
                                rdf['flag'] = True
                            elif obs[col]>obs['hi']:
                                rdf['limit'] = obs['hi']
                                rdf['flag'] = True
                        return pd.Series(rdf)


                    if metric_name not in self.base_data.columns:
                        print('Cannot find', metric_name)
                        return

                    # merge on INDUSTRY metric by region and complexity (.apply Overall if no match)
                    # This may need to be generalized if other filter methods are used for metric
                    IND_df = self.base_data[['TA_Region', 'TA_Complexity']].apply(assign_metric,
                                                                                  **{'data': metric_data},
                                                                                  axis=1)

                    IND_df = pd.concat([IND_df, self.base_data[metric_name]], axis=1)

                    # determine indices to flag
                    IND_flags = IND_df.apply(flag, **{'col':metric_name}, axis=1)
                    IND_flags = IND_flags[IND_flags['flag']==True]

                    # update columns
                    update_columns(IND_flags.index, metric_name, process, 'ids')

                    # update obs
                    update_obs(IND_flags.index, metric_name, process, IND_flags[['avg', 'limit']])

                    self.column_dict[metric_name]['Check_Count'] +=1
                    self.column_dict[metric_name]['Quality'] += obs-len(IND_flags)


                print('Checking against Industry Metrics')
                for metric_name, metric_dict in self.IND_METRICS.items():
                    metric_data = metric_dict['data']
                    #TODO: Adjust to include different metric types such as TRIR and Trips
                    if metric_dict['type']!='average':
                        print('Not an average. Skipping')
                        continue
                    if metric_name[0]=='X':
                        for t in ['A', 'P']:
                            metric_name = metric_name.replace('X', t)
                            check_ind_metrics(metric_name, metric_data)
                    else:
                        check_ind_metrics(metric_name, metric_data)

            process = 'ratio_cors'
            # calculates sense check from base_data and compares to defined bounds
            if process in check_list or 'ALL' in check_list:
                print('Checking against sense checks')
                for sc in self.sense_checks.keys():
                    self.base_data[sc] = self.sense_checks[sc].evaluate(self.base_data)
                    try:
                        sense_df = self.base_data[sc].apply(self.sense_checks[sc].flag)
                        sense_df = sense_df[sense_df['flag']==True]

                        # update columns
                        update_columns(sense_df.index, sc, process, 'ids')

                        # update obs
                        update_obs(sense_df.index, sc, process, sense_df[['limit']])

                        self.column_dict[sc]['Check_Count'] +=1
                        self.column_dict[sc]['Quality'] += obs-len(sense_df)
                    except Exception as e:
                        print(e)



            self.error_summary(show=True)

            process = 'Calculate Table Scoring'
            # calculates the missing rates for each table from table_dict
            for table, col_scores in self.table_dict.items():
                A, B = 0, 0
                for a,b in col_scores:
                    if a is not None:
                        A+=a
                    if b is not None:
                        B+=b
                self.table_dict[table] = A/B

            process = 'scoring'
            # Scores the diagnostic run
            print('Calculating Quality Scores')
            for col in self.column_dict.keys():
                if self.column_dict[col]['Check_Count']!=0:
                    self.column_dict[col]['Quality'] = self.column_dict[col]['Quality']/(obs*self.column_dict[col]['Check_Count'])
                else:
                    self.column_dict[col]['Quality'] = np.nan

            self.overall_score = pd.DataFrame.from_dict(self.column_dict, orient='index')['Quality'].mean()
            print(self.overall_score)
            print('-', time.strftime('%H:%M:%S',time.localtime()), 'Finished Diagnostics')
        except Exception as e:
            print()
            print(process)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            print()


    def get_column(self, col, sub_col=None):
        """
        Gets column from column diagnostic dictionary

        variables
        ---------
        col: string, column/variable of interest
        sub_col: string, error type to be investigated

        returns: pandas.DataFrame, dataframe version of sub dictionary of interest
        """

        if sub_col:
            return pd.DataFrame.from_dict(self.column_dict[col][sub_col])
        return(pd.DataFrame.from_dict(self.column_dict[col], orient='index'))


    def get_obs(self, obs):
        """
        Gets observation from observation diagnostic dictionary

        variables: col: string, observation/row of interest

        returns: pandas.DataFrame, dataframe version of observation dictionary of interest
        """
        return(pd.DataFrame.from_dict(self.obs_dict[obs], orient='index'))


    def get_duplicate(self, dup_number=0):
        """
        prints duplicate pair comparison of interest

        variable: dup_number: int, duplicate ID number
        """
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd.DataFrame.from_dict(self.comparison_set[dup_number]))


    def summary_report(self):
        """
        Prints summary data for:
            * column errors
            * duplicates
            * error types
            * data density
            * overall score
        """

        print('Overall Score:', self.overall_score)
        print()

        print('Data Size')
        print('Rows', len(self.base_data))
        print('Vars', len(self.base_data.columns))
        print('Size', self.base_data.size)
        print()

        print('Potential Duplicates', len(self.comparison_set))
        print()
        print('First 5 duplicates by Company, Facility, TA_Name, DC_Session', self.index)
        for i in range(5):
            if i ==len(self.comparison_set): break
            dup_dict = self.comparison_set[i]
            for v in dup_dict.values():
                print(v['Company'], v['Facility'], v['TA_Name'], v['DC_Session'], v[self.index])
            print()

        print()

        print('Error Types')
        error_dict = {}

        for i, es in self.obs_dict.items():
            for v in es.values():
                if type(v)==dict:
                    for e in v.keys():
                        if e == 'value':
                            continue
                        if e in error_dict.keys():
                            error_dict[e]+=1
                        else:
                            error_dict[e]=1

        err_df = pd.DataFrame.from_dict(error_dict, orient='index').sort_values(0, ascending=False)
        err_df['Total Pct']  = err_df[0]/self.base_data.size # under counts
        err_df['Pct of Errors'] = err_df[0]/err_df[0].sum()
        print(err_df)
        print()

        column_df = pd.DataFrame.from_dict(self.column_dict, orient='index').sort_values('Quality', ascending=True)

        print('Columns with no criteria')
        for c in sorted(column_df.index[column_df['Has_Criteria'].isnull()].tolist()):
            print('-',c)
        print()

        print('Missing Rate for TSE Tables')
        for table, rate in self.table_dict.items():
            if rate !=0:
                print(table, '\t\t', rate)
        print()

        print('Variable Quality Score Summary')
        def bin_quality_scores(x):
            for threshold in [0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999, 1]:
                if x <= threshold:
                    return threshold
        print(len(self.base_data.columns), 'variables')
        bins = column_df['Quality'].apply(bin_quality_scores)

        print(bins.value_counts(dropna=False).sort_index(ascending=False))
        print()

        print('25 Worst Performing columns')
        qual_scores = column_df['Quality'][(column_df['Has_Criteria'].notnull()) & (column_df['Quality']!=1)].head(25)
        # TDOD: Only show scores <50% of non-1 scores
        print(qual_scores)


    def error_summary(self, show=True):
        """
        Forms a summary of error types and prints the restuls
        """
        self.error_dict = {}
        for col, errs in self.column_dict.items():
            for error_type, errors in errs.items():
                if type(errors) in [dict, list]:
                    if len(errors)==0:
                        continue
                    if error_type not in self.error_dict.keys():
                        self.error_dict[error_type] = {}

                    for i, cnt in errors.items():
                        if type(cnt)==dict: #Scale or outlier error type
                            if col not in self.error_dict[error_type].keys():
                                self.error_dict[error_type][col]=0
                            self.error_dict[error_type][col]+=1
                        else:
                            if i in self.error_dict[error_type].keys():
                                self.error_dict[error_type][i] += cnt
                            else:
                                self.error_dict[error_type][i] = cnt

        if show:
            for error_type, error_dict in self.error_dict.items():
                print(error_type)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(pd.Series(error_dict).sort_values(ascending=False))
                print()


    def column_summary(self):
        """
        Forms a summary of the column diagnostics and prints the result
        """
        col_summary={}
        for col, errors in self.column_dict.items():
            col_summary[col]={}
            for error_type, vals in errors.items():
                if type(vals)==dict:
                    col_summary[col][error_type]=len(vals)
                else:
                    col_summary[col][error_type]=vals
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd.DataFrame.from_dict(col_summary, orient='index').sort_values('Quality'))


    def duplicate_summary(self):
        """
        forms a summary of duplciate observations and prints the result
        """
        dup_summary = {}
        for i, dup_dict in enumerate(self.comparison_set):
            k1 = list(dup_dict.keys())[0]
            v1 = dup_dict[k1]
            dup_summary[i] = {'Company': v1['Company'],'Facility': v1['Facility'], 'TA_Name': v1['TA_Name'], 'DC_Session': v1['DC_Session']}
            if self.index == 'Survey_Unit_ID':
                dup_summary[i]['Unit_Name_APN_Std'] = v1['Unit_Name_APN_Std']
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd.DataFrame.from_dict(dup_summary, orient='index'))


    @logErrors()
    def export_report(self, export_path=''):
        """
        Exports diagnostic attributes to json files

        variable: export_path: string, export_path to save results of the diagnostics
        """


        print('EXPORT DIAGNOSTICS')
        print('Checks if path exists')
        if not os.path.isdir(os.path.join(export_path)):
            os.mkdir(os.path.join(export_path))

        print('Base Data')
        with open(os.path.join(export_path, 'Base_data.json'), 'w') as f:
            json.dump(self.base_data.to_json(orient='split'), f)

        print('var diagnostics')
        with open(os.path.join(export_path, 'variable_diagnostics.json') , 'w') as f:
            json.dump(self.column_dict, f, cls=NpEncoder)
        print('obs diagnostics')
        with open(os.path.join(export_path, 'observation_diagnostics.json'), 'w') as f:
            json.dump(self.obs_dict, f, cls=NpEncoder)
        print('TAN values')
        with open(os.path.join(export_path, 'TAN_value_options.json'), 'w') as f:
            json.dump(self.TAN_only_list, f, cls=NpEncoder)
        print('Criteria')
        with open(os.path.join(export_path, 'Diagnostic_criteria.json'), 'w') as f:
            json.dump(self.criteria_dict, f)
        print('badData')
        with open(os.path.join(export_path, 'labelled_badData.json'), 'w') as f:
            json.dump(self.labelled_badData, f)
        print('Table diagnostics')
        with open(os.path.join(export_path, 'table_diagnostics.json'), 'w') as f:
            json.dump(self.table_dict, f, cls=NpEncoder)
        print('duplicates_check')
        with open(os.path.join(export_path, 'duplicate_check.json'), 'w') as f:
            json.dump(self.comparison_set, f, cls=NpEncoder)

        print('Duplicate review report')
        with pd.ExcelWriter(os.path.join(export_path, 'Duplicates_to_review.xlsx')) as writer:
            for couple in self.comparison_set:
                tab_name = [str(x) for x in couple.keys()]
                pd.DataFrame(couple).to_excel(writer, sheet_name='_'.join(tab_name))

        diagnostic_path = r'C:\Users\{}\AP-Networks\Benchmarking Group - Documents\Data Cleaning\Diagnostics'.format(username)

        print('Copy Notebook to', os.path.join(export_path, 'Validation_Notebook.ipynb'))
        copy(os.path.join(diagnostic_path, 'Validation_Notebook.ipynb'), os.path.join(export_path, 'Validation_Notebook.ipynb'))

        print('Exported to JSON')


    def confirm_reviewed(self, obs=None, col_error=None, comments='', debug=False):
        """
        updates the self.review dictionary with the observation, columns, and errors to be marked as OK.
        The command should be run for a group of observations that shared similar column_error combinations,
        a single observation, or to mark all observations with the given col_error combinations

        obs: list of observations that can be marked as OK
            if obs is None, all cols:[errors] will be marked as OK
        col_error: dictionary of columns and errors that can be cleared for the observation
            example: col: [errors]. either col or errors can be set to 'all'. If col_error is None,
            the entire observation will be set as OK
        comments: str, reviewers comment explaining why this is being marked as OK

        """
        def format_col_error(col_error):
            """
            formats the col_error variable into a dictionary. Fills in 'all'
            when appropriate
            """
            if type(col_error)==str:
                if col_error in self.column_dict.keys():
                    tmp_col_error = {col_error: 'all'}
                elif col_error in self.error_dict.keys():
                    tmp_col_error = {'all': col_error}
                else:
                    print('Cannot find col_error in error types or columns')
                    return None
            elif type(col_error)==list:
                tmp_col_error = {}
                for ce in col_error:
                    if ce in self.column_dict.keys():
                        tmp_col_error[ce] = 'all'
                    elif ce in self.error_dict.keys():
                        tmp_col_error['all'] = ce
            else:
                return col_error

            return tmp_col_error


        if obs is None and col_error is None:
            print('No data provided to update')
            return

        if not hasattr(self, 'reviewed'):
            self.reviewed = []

        entry = {'index': None,
                 'error_type': None,
                 'column': None,
                 'data_value': None,
                 'diag_value': None,
                 'comments': comments}

        if obs is not None and col_error is not None:
            if type(obs)!=list:
                obs = [obs]

            col_error = format_col_error(col_error)
            if col_error is None:
                return

            for o in obs:
                if debug: print(o)
                entry['index'] = o
                for ID in self.obs_dict[o].keys():
                    if re.search('_ID', ID):
                        entry[ID] = self.obs_dict[o][ID]

                if 'all' in col_error.keys():
                    for c in self.obs_dict[o].keys():
                        if debug: print(c)
                        if re.search('ID', c):
                            continue
                        entry['column'] = c
                        entry['data_value'] = self.obs_dict[o][c]['value']

                        if type(col_error['all'])!=list:
                            col_error['all'] = [col_error['all']]
                        for e in col_error['all']:
                            if debug: print(e)
                            if e in self.obs_dict[o][c].keys():
                                entry['error_type'] = e
                                entry['diag_value'] = self.obs_dict[o][c][e]
                                self.reviewed.append(deepcopy(entry))

                else:
                    for c in col_error.keys():
                        if debug: print(c)
                        entry['column'] = c
                        entry['data_value'] = self.obs_dict[o][c]['value']

                        if type(col_error[c])!=list:
                            col_error[c] = [col_error[c]]

                        if 'all' in col_error[c]:
                            for e in self.obs_dict[o][c].keys():
                                if e =='value':
                                    continue
                                elif e in self.obs_dict[o][c].keys():
                                    entry['error_type'] = e
                                    entry['diag_value'] = self.obs_dict[o][c][e]
                                    self.reviewed.append(deepcopy(entry))

                        else:
                            for e in col_error[c]:
                                if e in self.obs_dict[o][c].keys():
                                    entry['error_type'] = e
                                    entry['diag_value'] = self.obs_dict[o][c][e]
                                    self.reviewed.append(deepcopy(entry))


        elif obs is not None and col_error is None:
            for o in obs:
                if debug: print(o)
                entry['index'] = o
                for ID in self.obs_dict[o].keys():
                    if re.search('_ID', ID):
                        entry[ID] = self.obs_dict[o][ID]

                for c in self.obs_dict[o].keys():
                    if debug: print(c)
                    if re.search('ID', c):
                        continue
                    entry['column'] = c
                    entry['data_value'] = self.obs_dict[o][c]['value']

                    for e in self.obs_dict[o][c].keys():
                        if debug: print(e)
                        if e =='value':
                            continue
                        entry['error_type'] = e
                        entry['diag_value'] = self.obs_dict[o][c][e]
                        self.reviewed.append(deepcopy(entry))
        elif obs is None and col_error is not None:
            col_error = format_col_error(col_error)
            if col_error is None:
                return

            for o in self.obs_dict.keys():
                if debug: print(o)
                entry['index'] = o
                for ID in self.obs_dict[o].keys():
                    if re.search('_ID', ID):
                        entry[ID] = self.obs_dict[o][ID]

                if 'all' in col_error.keys():
                    for c in self.obs_dict[o].keys():
                        if debug: print(c)
                        if re.search('ID', c):
                            continue
                        entry['column'] = c
                        entry['data_value'] = self.obs_dict[o][c]['value']

                        if type(col_error['all'])!=list:
                            col_error['all'] = [col_error['all']]
                        for e in col_error['all']:
                            if debug: print(e)
                            if e in self.obs_dict[o][c].keys():
                                entry['error_type'] = e
                                entry['diag_value'] = self.obs_dict[o][c][e]
                                self.reviewed.append(deepcopy(entry))

                else:
                    for c in col_error.keys():
                        entry['column'] = c
                        entry['data_value'] = self.obs_dict[o][c]['value']

                        if type(col_error[c])!=col_error[c]:
                            col_error[c] = [col_error[c]]

                        if 'all' in col_error[c]:
                            for e in self.obs_dict[o][c].keys():
                                if e =='value':
                                    continue
                                entry['error_type'] = e
                                entry['diag_value'] = self.obs_dict[o][c][e]
                                self.reviewed.append(deepcopy(entry))
                        else:
                            for e in col_error[c]:
                                if e in self.obs_dict[o][c].keys():
                                    entry['error_type'] = e
                                    entry['diag_value'] = self.obs_dict[o][c][e]
                                    self.reviewed.append(deepcopy(entry))


    def export_reviewed(self, export_path=''):
        """
        Exports the self.reviewed dict to folder
        """
        print('export updated reviewed diagnostics')
        with open(os.path.join(export_path, 'reviewed.json'), 'w') as f:
            json.dump(self.reviewed, f, cls=NpEncoder)


