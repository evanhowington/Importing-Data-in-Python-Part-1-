"""
Listing sheets in Excel files
"""
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xl
xl = pd.ExcelFile(file)
# Print sheet names
print(xl.sheet_names)

"""
Importing sheets from Excel files
"""

# Load a sheet into a DataFrame by name : df1
df1 = xl.parse('2004')
# Print the head of the DataFrame df1
print(df1.head())
# Load a sheet into a DataFrame by index: df2
df2 = xl.parse('2002', index_col=0)
# Print the head of the DataFrame df2
print(df2.head())

"""
importing SAS files
"""
import matplotlib.pyplot as dick
import pandas as pd
# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
dick.ylabel('count')
dick.show()

"""
importing STATA files
"""

# import pandas
import pandas as pd

# load stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')
# print the head of the DataFrame df
print(df.head())
# plot histogram of one column of the DataFrame
import matplotlib.pyplot as plt

pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of desease')
plt.ylabel('Number of coutnries')
plt.show()

"""
importing HDF5 files
"""
# import packages
import numpy as np
import h5py

# assign filename: file
file = 'LIGO_data.hdf5'

# load file: data
data = h5py.File(file, 'r')
# print the datatype of the loaded file
print(type(data))
# print the keys of the file
for key in data.keys():
    print(key)

"""
Extracting data from your HDF5 file
"""
import numpy as np
import matplotlib.pyplot as plt
# Get the HDF5 group: group
group = data['strain']
# data was loaded in the previous section of code
# Check out keys of group
for key in group.keys():
    print(key)
# set the variable equal to time series data: strain
strain = data['strain']['Strain'].value
# set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0 , 1 , 1/num_samples)
# plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()
