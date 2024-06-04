"""
Author: Brian Riggs
Version: 2
Date: 20240529

Class structure for sense_checks to use in diagnostics. Each object
contains the function and function variables necessary to calculate the sense
check variable. The object also includes the bounds as determined by adjusted
box plot IQR measure (https://en.wikipedia.org/wiki/Box_plot#Variations). The
bounds can be manually set as well.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.stattools import medcouple
from typing import Union

class sense_check():
    """
    Class to store sense checks which executes a function and compares result to
    threshold values.

    Attributes
    ----------
    func: function, function used to calculate the sense check metric typically
        arguments will be in the form (data, **variables in data) and return a single
        value
    func_kwargs: dict, {variable: name} to be used in the function
    bounds: tuple, bounds for the sense check metric. Typically a sorted numeric tuple
    gen_source: object, data object used to generate bounds if bounds are not provided


    Functions
    ---------
    __init__: initializes instance of sense_check
    display: displays all attributes of the sense_check
    evaluate: uses the sense_check function to evaluate data using func_kwargs
    flag: compares sense_check bound to a value. Returns True if value is outside
        of bounds
    """

    def __init__(self, func, func_kwargs, bounds=None, gen_source=None):
        """
        initializes instance of sense_check

        arguments
        ---------
        self
        see attributes

        Returns
        -------
        none
        """
        if bounds is None and gen_source is None:
            raise ValueError('Need some type of criteria (bounds or gen_source) for bounds')
        elif bounds is not None and gen_source is not None:
            raise ValueError('Provide only one type of criteria (bounds or gen_source) for bounds')
        else:
            self.func = func
            self.func_kwargs = func_kwargs
            if bounds is None and gen_source is not None:
                # outlier check
                column_data = self.evaluate(gen_source)
                non_nans = [x for x in column_data.tolist() if not np.isnan(x)]

                p25 = column_data.quantile(q=0.25)
                p75 = column_data.quantile(q=0.75)

                mc = medcouple(non_nans)

                if mc >=0:
                    lower = p25-1.5*np.exp(-4*mc)*abs(p75-p25)
                    upper = p75+1.5*np.exp(3*mc)*abs(p75-p25)
                else:
                    lower = p25-1.5*np.exp(-3*mc)*abs(p75-p25)
                    upper = p75+1.5*np.exp(4*mc)*abs(p75-p25)

                self.bounds=(lower, upper)
            else:
                self.bounds= bounds


    def display(self):
        """
        Prints attributes of sense_check class

        Arguments
        ---------
        self

        Returns
        -------
        None
        """
        attrs = vars(self)
        for item in attrs.items():
            print("%s: %s" % item)
        print(self.func.__doc__)

    def __str__(self):
        return '\n'.join(["%s: %s" % item for item in vars(self).items()])


    def evaluate(self, data):
        """
        Evaluates the self.func using self.func_kwargs

        data: pandas dataframe or dict

        returns single value or series using sense_check function
        """
        if isinstance(data, pd.DataFrame):
            return data.apply(self.func, **self.func_kwargs, axis=1)
        elif isinstance(data, pd.Series):
            return self.func(data, **self.func_kwargs)
        elif isinstance(data, dict):
            for k,v in self.func_kwargs.items():
                if v not in data.keys():
                    break
            return self.func(data, **self.func_kwargs)
        else:
            print('No evaluation. Data in wrong type')
            return None


    def flag(self, x):
        """
        compares sense_check bound to a value. Returns True if value is outside
            of bounds

        x: object, value to be compared

        returns: pd.Series, includes the bound exceeded and a truth value.
        """
        
        if self.bounds[0] and self.bounds[1]:
            if x<self.bounds[0]:
                return False
            elif x>self.bounds[1]:
                return False
        return True
        


def ratio_xy(data, x, y):
    try:
        return(data[x]/data[y])
    except:
        return None

def sum_total(data:Union[pd.DataFrame, pd.Series], total_name, component_names):
    '''
    compares a data that should be a total against its components. Bounds
    should be reflective of an appropriate order of magnitude. 

    e.g. if sum is 10000, the bounds should be on the order of 10
    '''
    try:
        return data[total_name] - data[component_names].sum()
    except:
        return None