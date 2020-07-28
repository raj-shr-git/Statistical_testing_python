import numpy as np
import pandas as pd
from scipy.stats import norm as z_nm
    
def one_porportion_ztest(smp_porportion,pop_proportion,alternative='two-sided',nan_policy=False):
    """
    Description :: This function is created for performing the one population proportion Z test.

    Input Parameters :: It accepts below parameters:
        1. smp_proportion : It is the sample proportion array. Sample proportion mean is calculated form this array.
                            The number of values in this array are considered as n i.e. number of samples. 
        2. pop_proportion : It is the propotion which is currently accepted as the Null Hypothesis.
        3. alternative : This parameter represnts the kind of tail test that you can perform. It expects below values:
                            a) "smaller" : Executing the left tail test
                            b) "larger" : Executing the right tail test
                            c) "two-sided" : Executing the both tail test and it is also the default one.
        4. nan_policy : This parameter represnts the kind of Nulls filling you want to opt. It expects below values:
                            a) "mean" : Replace Null values with mean of the sample proportion array
                            b) "median" : Replace Null values with median of the sample proportion array
                            c) If given any other value then all Null's, NaN's and None's will be dropped from the sample proportion array.     

    Return :: It returns below two values:
                1. Test Statistic
                2. P-value of the Test Statistic    
    """

    if nan_policy == 'mean':
        smp_prop = np.array(pd.Series(smp_proportion).fillna(np.mean(smp_proportion)))
    elif nan_policy == 'median':
        smp_prop = np.array(pd.Series(smp_proportion).fillna(np.median(smp_proportion)))
    else:
        smp_prop = np.array(pd.Series(smp_porportion).dropna())

    smp_prop_mean = np.mean(smp_prop, dtype=np.float)
    n = len(smp_prop)

    pops_diff = (smp_prop_mean - pop_proportion)
    denom = np.sqrt((pop_proportion * (1-pop_proportion))/n)

    test_stat = np.divide(pops_diff,denom)

    if alternative == 'smaller':
        p_val = z_nm.cdf(test_stat)
    elif alternative == 'larger':
        p_val = z_nm.sf(test_stat)
    elif alternative == 'two-sided':
        test_stat_abs = np.abs(test_stat)
        p_val = 2 * z_nm.sf(test_stat_abs)

    test_results = (test_stat, p_val)
    return test_results