import schema
import os
import pandas as pd
import run_diagnostics

from diagnostic import diagnostics

print('test import')
schema_dict = schema.import_schema(os.path.join(os.getcwd(), 'test_data'), 'test_schema.xlsx')
test_data = pd.read_excel(os.path.join(os.getcwd(), 'test_data', 'test_table.xlsx'))

x = diagnostics(test_data, schema_dict, 'A')
x.run_diagnostics(name_columns='ID')

#print(x.column_dict)
print(x.comparison_set)
#run_diagnostics.enforce_schema(test_data, schema_dict)





