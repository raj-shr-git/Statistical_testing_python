# Importing Packages
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
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
    else:
        raise ValueError('invalid alternative')

    test_results = (test_stat, p_val)
    return test_results

def hyp_test(test_stats_result,loc,kind_of_test=['l','r','lr']):
    """
    Description: This function is performing the p-value comparison with the level of significance or alpha. It checks below condition:
                    1. p <= alpha
                    
    Input Parameters: It accepts below inputs:
                    1. test_stats_result : Values of Test Statistic and p-value
                    2. loc : Level of confidence 
                    3. kind_of_test : Any value from the options -- ['l','r','lr']
                    
    Return: Print the comparison result b/w p-value and alpha.
    """
    alpha = 1 - loc
    if (kind_of_test == 'l') & (test_stats_result[0] < 0) & (test_stats_result[1] < alpha):
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    elif (kind_of_test == 'r') & (test_stats_result[0] > 0) & (test_stats_result[1] < alpha):
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    elif (kind_of_test == 'lr') & (test_stats_result[1] < alpha):    
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    else:
        print("Researcher claim is wrong. Thus, fail to reject the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))

def chi_square_one_pop(f_exp,loc,test_tail=(['l','r','lr']),ddof=False,sample_data=False,sample_stddev=False):
    """
    Description: This function performs the Chi-square test for one population.
    
    Input parameters: It accepts below parameters:
        1. f_exp : Expected Standard Deviation value or Standard Deviation value from population
        2. loc : Level of confidence. Used this parameter to calculate the critical value
        3. test_tail : This parameter represents which kind of test you want to perform
        4. ddof : Degree of freedom. Use this paramter if you want to provide the adhoc value of dof
        5. sample_data : One dimensional array containing the sample data
        6. sample_stddev : You can provide the standard deviation of the sample directly as an input to perform the chi-sqaure test
        
    Returns:
        - In case of 'l' means left-tail test, it returns,
            - test_statistic
            - left critical value
            - p_value
        
        - In case of 'r' means right-tail test, it returns,
            - test_statistic
            - right critical value
            - p_value

        - In case of 'lr' means both-tail test, it returns,
            - test_statistic
            - left critical value
            - right critical value
            - p-value based on left tail
            - p-value based on right tail
"""
    def cal_mean_var_std(input_array):
        """
        Description: This function calculates the mean, variance and standard deviation of the 1 population.
        
        Input: It accepts below inp parameters:
            1. input_array : Sample Population-1
            
        Returns: Population-1:
                    - Mean
                    - Variance
                    - Standard Deviation
        """
        sample_data_mean = round(np.mean(input_array,dtype='float'),3)
        sample_data_stddev = round(np.mean(input_array,dtype='float'),3)
        sample_data_var = round(np.var(input_array,dtype='float'),3)
        return sample_data_mean, sample_data_var, sample_data_stddev
    
    if sample_stddev != False and sample_data == False and ddof != False:
        sample_data_stddev = sample_stddev
        sample_data_var = sample_data_stddev**2
        dof = ddof
    elif sample_data != False and sample_stddev == False and ddof == False:
        sample_data_mean, sample_data_stddev, sample_data_var = cal_mean_var_std(sample_data)
        total_obs = len(sample_data)
        dof = (total_obs-1)
    elif sample_data != False and sample_stddev == False and ddof != False:
        sample_data_mean, sample_data_stddev, sample_data_var = cal_mean_var_std(sample_data)
        dof = ddof
    
    f_exp_var = f_exp**2
    
    test_stat = (dof*sample_data_var)/f_exp_var
    
    def left_tail_crit_p_val(tail_test,c,df,test_statistic):
        """
        Description: This function is performing the left tail chi-square hypothesis testing.
        
        Input: It accepts below input parameters:
            1. tail_test : This should be 'l' as we are performing left-tail test
            2. c : Level of confidence
            3. df : Degree of freedom
            4. test_statistic : Test statistic that we have calculated from the sample data
        
        Returns: Left tail-test:
                    - Test Statistic
                    - Critical value
                    - p_value
        """
        alpha = 1 - c
        lower_tail_prob = alpha
        critical_val = scipy_stats.chi2.ppf(lower_tail_prob,dof)
        p_value = 1 - scipy_stats.chi2.cdf(test_stat,dof)
        return test_stat, critical_val, p_value
    
    def right_tail_crit_p_val(tail_test,c,df,test_statistic):
        """
        Description: This function is performing the right tail chi-square hypothesis testing.
        
        Input: It accepts below input parameters:
            1. tail_test : This should be 'r' as we are performing left-tail test
            2. c : Level of confidence
            3. df : Degree of freedom
            4. test_statistic : Test statistic that we have calculated from the sample data
        
        Returns: Right tail-test:
                    - Test Statistic
                    - Critical value
                    - p_value
        """
        lower_tail_prob = c
        critical_val = scipy_stats.chi2.ppf(lower_tail_prob,dof)
        p_value = 1 - scipy_stats.chi2.cdf(test_stat,dof)
        return test_stat, critical_val, p_value
         
    def two_tail_crit_p_val(tail_test,c,df,test_statistic):
        """
        Description: This function is performing the both or two tail chi-square hypothesis testing.
        
        Input: It accepts below input parameters:
            1. tail_test : This should be 'lr' as we are performing left-tail test
            2. c : Level of confidence
            3. df : Degree of freedom
            4. test_statistic : Test statistic that we have calculated from the sample data
        
        Returns: Both or Two tail-test:
                    - Test Statistic
                    - Left critical value
                    - Right critical value
                    - p value
        """  
        l_alpha_by_2 = (1 - c)/2
        l_lower_tail_prob = l_alpha_by_2
        l_critical_val = scipy_stats.chi2.ppf(l_lower_tail_prob,dof)
        r_lower_tail_prob = (1 - l_alpha_by_2)
        r_critical_val = scipy_stats.chi2.ppf(r_lower_tail_prob,dof)
        p_value = 1 - scipy_stats.chi2.cdf(test_stat,dof)
        return test_stat, l_critical_val, r_critical_val, p_value
    
    if test_tail == 'l':
        test_stat, l_cri_val, p_value = left_tail_crit_p_val(tail_test=test_tail,c=loc,df=dof,test_statistic=test_stat)
        return test_stat, l_cri_val, p_value
    if test_tail == 'r':
        test_stat, r_cri_val, p_value = right_tail_crit_p_val(tail_test=test_tail,c=loc,df=dof,test_statistic=test_stat)
        return test_stat, r_cri_val, p_value
    if test_tail == 'lr':    
        test_stat, l_cric_val, r_cric_val, p_value = two_tail_crit_p_val(tail_test=test_tail,c=loc,df=dof,test_statistic=test_stat)
        return test_stat, l_cric_val, r_cric_val, p_value
        
def chi_square_hyp_test(test_stats_result,loc,kind_of_test=['l','r','lr']):
    """
    Description: This function is performing the p-value comparison(only for chi square distribution) with the level of significance or alpha. It checks below condition:
                    1. p <= alpha
                    
    Input Parameters: It accepts below inputs:
                    1. test_stats_result : Values of Test Statistic, Left critical value, Right critical value and p-value
                    2. loc : Level of confidence 
                    3. kind_of_test : Any value from the options -- ['l','r','lr']
                    
    Return: Print the comparison result b/w p-value and alpha.
    """
    alpha = 1 - loc
    if (kind_of_test == 'l') & (test_stats_result[-1] < alpha):
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    elif (kind_of_test == 'r') & (test_stats_result[-1] < alpha):
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    elif (kind_of_test == 'lr') & (test_stats_result[-1] < alpha) & ((test_stats_result[0] < test_stats_result[1]) or (test_stats_result[0]) > test_stats_result[2]):    
        print("Researcher claim is right. Thus, rejected the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
    else:
        print("Researcher claim is wrong. Thus, fail to reject the Null Hypothesis at {} L.O.C and {} L.O.S".format(loc,round(alpha,2)))
        
def ztest_notpooled(x1, x2=None, value=0, alternative='two-sided', usevar='notpooled', ddof=1.):
    '''test for mean based on normal distribution, one or two samples

    In the case of two samples, the samples are assumed to be independent.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples
    x2 : array_like, 1-D or 2-D
        second of the two independent samples
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: difference in means not equal to value (default)
           'larger' :   H1: difference in means larger than value
           'smaller' :  H1: difference in means smaller than value

    usevar : string, 'notpooled'
        ``notpooled``, means the standard deviation of the samples is assumed to be
        not the same.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Returns
    -------
    tstat : float
        test statisic
    pvalue : float
        pvalue of the t-test
    '''
    def zstat_gen(value1, value2, std_diff, alternative, diff=0):
        '''generic (normal) z-test to save typing
        can be used as ztest based on summary statistics
        '''
        zstat = (value1 - value2 - diff) / std_diff
        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = scipy_stats.norm.sf(np.abs(zstat))*2
        elif alternative in ['larger', 'l']:
            pvalue = scipy_stats.norm.sf(zstat)
        elif alternative in ['smaller', 's']:
            pvalue = scipy_stats.norm.cdf(zstat)
        else:
            raise ValueError('invalid alternative')
        return zstat, pvalue
    
    if usevar != 'notpooled':
        #print("You are using Two populations whose variances are assumed to be ``Unequal`` or ``Not-pooled``")
        raise NotImplementedError('only usevar="not-pooled" is implemented')

    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)
    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        var_not_pooled = ((x1_var/nobs1) + (x2_var/nobs2))
    else:
        var_not_pooled = x1_var / (nobs1 - ddof)
        x2_mean = 0

    std_diff = np.sqrt(var_not_pooled)
    z_stat, pvalue = zstat_gen(x1_mean, x2_mean, std_diff, alternative, diff=value)
    return z_stat, pvalue

def f_dist_test(x1,x2,loc=0.95,alternative='two-sided',usevar='pooled',cal_x1_x2_var=False,n1_obsv=False,n2_obsv=False):
    """
    Description : This function is performing the f-test for the pooled populations.
    
    Input Parameters : It accepts below inputs:
                        1. x1 : Population one variance (float or int)
                        2. x2 : Population two variance (float or int)
                        3. loc : Level of confidence (by default 0.95 or 95%)
                        4. alternative : Kind of tail test either ('larger', 'smaller' or 'two-sided')
                        5. uservar : string, 'pooled'
                            ``notpooled``, means the standard deviation of the samples is assumed to be same.
                        6. cal_x1_x2_var : Flag for whether variance of both the populations to be calculated from the given arrays
                            ``False`` : Means variance of pop1 and pop2 are given in x1 and x2
                            Other than False : Means x1 and x2 are the arrays and variances to be computed from the same
                        7. n1_obsv : Sample 1 size (int)
                        8. n2_obsv : Sample 2 size (int)
    Return :
    -------
    f_test_stat : float
        test statisic
    p_value : float
        p-value of the f-test
    f_critical1 : float
        First tail critical value
    f_critical2 : float
        Second tail critical value
    """
    if cal_x1_x2_var == False:
        s1_var = (1. * x1)
        s2_var = (1. * x2)
        nobs1 = n1_obsv
        nobs2 = n2_obsv
        dof1 = nobs1 - 1
        dof2 = nobs2 - 1
    else:
        s1,s2 = np.asarray(x1),np.asarray(x2)
        nobs1 = len(s1)
        nobs2 = len(s2)
        dof1 = nobs1 - 1
        dof2 = nobs2 - 1
        s1_var,s2_var = np.var(s1),np.var(s2)
        
    f_test_stat = (s1_var) * (1./s2_var)

    if alternative == 'larger':
        f_distribution = scipy_stats.f(dof1,dof2)
        p_value = 1 - f_distribution.cdf(f_test_stat)
        f_critical = f_distribution.ppf(loc)
        return f_test_stat, p_value, f_critical
    elif alternative == 'smaller':
        f_distribution = scipy_stats.f(dof1,dof2)
        p_value = f_distribution.cdf(f_test_stat)
        f_critical = f_distribution.ppf(1 - loc)
        return f_test_stat, p_value, f_critical
    elif alternative == 'two-sided':
        alpha = 1 - loc
        alpha_by_2 = alpha/2
        one_minus_alpha_by_2 = 1 - alpha_by_2
        f_distribution = scipy_stats.f(dof2,dof1)
        p_value = 2 * min(f_distribution.cdf(f_test_stat), 1 - f_distribution.cdf(f_test_stat))
        f_critical1 = f_distribution.ppf(alpha_by_2)
        f_critical2 = f_distribution.ppf(one_minus_alpha_by_2)
        return f_test_stat, p_value, f_critical1, f_critical2
    else:
        raise ValueError('invalid alternative')