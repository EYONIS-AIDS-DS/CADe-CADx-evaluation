import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


import config_paper as config
from evaluate_common.logger import logger


def plot_distribution_bootstrap(AUC_arr, AUC_arr_2, name_comparison, expdir):
    fig = plt.figure()
    bins = np.linspace(min(min(AUC_arr),min(AUC_arr_2)), max(max(AUC_arr), max(AUC_arr_2)), 50)
    plt.title('Distribution of bootstrap values ' + name_comparison)
    plt.hist(AUC_arr, bins, alpha=0.5, label= "distribution 1")
    plt.hist(AUC_arr_2, bins, alpha=0.5, label= "distribution 2")
    plt.legend(loc='upper right')
    plt.ylabel('Occurence')
    
    plt.savefig(expdir.joinpath("distribution_BOOTSTRAPP_"+name_comparison+".svg"))
    plt.savefig(expdir.joinpath("distribution_BOOTSTRAPP_"+name_comparison+".png"))
    plt.close(fig)
    

def statistical_test_main(list_of_tuples_of_pairs_of_bootstrap_paths, expdir_analysis,):

    # Welch t-test 
    # t-test unpaired two sided unequal mean: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    # H1 is arr_1  greater than arr_2
    
    # input is structured as list of tuples of paths like [("name_comparison 1", "path1", "path2"), ("name_comparison 2", "path3", "path4")]
    # of course tuple are ordered and the order of the H1 test is "path1"  greater than "path2"
    
    
    for tuple_of_pairs in list_of_tuples_of_pairs_of_bootstrap_paths:

        arr_1 = np.load(tuple_of_pairs[1])
        arr_2 = np.load(tuple_of_pairs[2])
                
        # if arr_1 is a vector 
        if arr_1.ndim == 1:
            list_of_test_name = []
            list_of_pvalue_equal = []
            list_of_pvalue_unequal = []
            list_test_result_equal = []
            list_test_result_unequal = []
            plot_distribution_bootstrap(arr_1, arr_2, tuple_of_pairs[0], expdir_analysis)
            statistic, pvalue_unequal = stats.ttest_ind(arr_1, arr_2, alternative= "greater",  equal_var=False) 
            statistic, pvalue_equal = stats.ttest_ind(arr_1, arr_2, alternative= "greater",  equal_var=True)    

            logger.info("##################    TEST " + str(tuple_of_pairs[0])+ "    ##################")
            logger.info(f"The P_valule of t-test unpaired one sided (1>2) unequal variance is: {pvalue_unequal}")
            logger.info(f"The P_valule of t-test unpaired one sided (1>2) equal variance is: {pvalue_equal}")

            list_of_test_name.append(tuple_of_pairs[0])           
            list_of_pvalue_equal.append(pvalue_equal)
            list_of_pvalue_unequal.append(pvalue_unequal)
            # https://www.jcpcarchives.org/full/p-value-statistical-significance-and-clinical-significance-121.php
            if pvalue_equal < 0.0001:
                logger.info("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
            elif pvalue_equal < 0.001:
                logger.info("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
            elif pvalue_equal < 0.01:
                logger.info("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
            elif pvalue_equal < 0.05:
                logger.info("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
            elif pvalue_equal < 0.1:
                logger.info("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
            elif pvalue_equal >= 0.1:
                logger.info("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                list_test_result_equal.append("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")

            if pvalue_unequal < 0.0001:
                logger.info("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
            elif pvalue_unequal < 0.001:
                logger.info("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
            elif pvalue_unequal < 0.01:
                logger.info("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
            elif pvalue_unequal < 0.05:
                logger.info("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
            elif pvalue_unequal < 0.1:
                logger.info("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
            elif pvalue_unequal >= 0.1:
                logger.info("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                list_test_result_unequal.append("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
            test_results = pd.DataFrame(list_of_test_name, columns=["test_name"])
            test_results["p_value equal variance" ] = list_of_pvalue_equal
            test_results["test_result equal variance"] = list_test_result_equal
            test_results["p_value unequal variance" ] = list_of_pvalue_unequal
            test_results["test_result unequal variance"] = list_test_result_unequal
            test_results.to_csv(os.path.join(expdir_analysis,"test_results_"+ tuple_of_pairs[0]+ ".csv",), index=False, sep=",",)
                
        else:
            for i in range(arr_1.shape[1]):
                # put in array_1 and array_2 the ith column of the matrix tuple_of_pairs[1] and tuple_of_pairs[2]
                list_of_test_name = []
                list_of_pvalue_unequal = []
                list_test_result_unequal = []
                list_of_pvalue_equal = []
                list_test_result_equal = []
                array_1 = arr_1[:,i]
                array_2 = arr_2[:,i]
                plot_distribution_bootstrap(array_1, array_2, tuple_of_pairs[0]+"_op__" +str(i+1), expdir_analysis)
                statistic, pvalue_unequal = stats.ttest_ind(array_1, array_2, alternative= "greater",  equal_var=False)   
                statistic, pvalue_equal = stats.ttest_ind(array_1, array_2, alternative= "greater",  equal_var=True)                    
                logger.info("##################    TEST " + str(tuple_of_pairs[0])+"_op_"+str(i+1)+ "    ##################")
                logger.info(f"The P_valule of t-test unpaired one sided (1>2) unequal variance is: {pvalue_equal}")
                logger.info(f"The P_valule of t-test unpaired one sided (1>2) equal variance is: {pvalue_unequal}")
                list_of_test_name.append(tuple_of_pairs[0]+"_op_"+str(i+1))
                list_of_pvalue_equal.append(pvalue_equal)
                list_of_pvalue_unequal.append(pvalue_unequal)
                # https://www.jcpcarchives.org/full/p-value-statistical-significance-and-clinical-significance-121.php
                if pvalue_unequal < 0.0001:
                    logger.info("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                elif pvalue_unequal < 0.001:
                    logger.info("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                elif pvalue_unequal < 0.01:
                    logger.info("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                elif pvalue_unequal < 0.05:
                    logger.info("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                elif pvalue_unequal < 0.1:
                    logger.info("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                elif pvalue_unequal >= 0.1:
                    logger.info("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                    list_test_result_unequal.append("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                else:
                    logger.info("missing analysis")
                    list_test_result_unequal.append("missing analysis")

                if pvalue_equal < 0.0001:
                    logger.info("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis can be rejected at 0.0001 level of significance. extremely strong evidence against the null hypothesis")
                elif pvalue_equal < 0.001:
                    logger.info("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis can be rejected at 0.001 level of significance. Very strong evidence against the null hypothesis")
                elif pvalue_equal < 0.01:
                    logger.info("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis can be rejected at 0.01 level of significance. Strong evidence against the null hypothesis")
                elif pvalue_equal < 0.05:
                    logger.info("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis can be rejected at 0.05 level of significance. Moderate evidence against the null hypothesis")
                elif pvalue_unequal < 0.1:
                    logger.info("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis can be rejected at 0.1 level of significance.  Weak evidence against the null hypothesis")
                elif pvalue_equal >= 0.1:
                    logger.info("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                    list_test_result_equal.append("The null hypothesis cannot be rejected at 0.1 level of significance. No evidence against the null hypothesis")
                else:
                    logger.info("missing analysis")
                    list_test_result_equal.append("missing analysis")


                test_results = pd.DataFrame(list_of_test_name, columns=["test_name"])
                test_results["p_value equal variance" ] = list_of_pvalue_equal
                test_results["test_result equal variance"] = list_test_result_equal
                test_results["p_value unequal variance" ] = list_of_pvalue_unequal
                test_results["test_result unequal variance"] = list_test_result_unequal
                test_results.to_csv(os.path.join(expdir_analysis,"test_results_"+ tuple_of_pairs[0]+"_op_"+str(i+1)+ ".csv",), index=False, sep=",",)
            
        
           
        
        
# #########################################################################
# #########################################################################
# ######          ARGUMENTS of the PROGRAM               ##################
# #########################################################################
# #########################################################################


def parse_args(args):
    """
    Checks and parse args used
    """
    parser = argparse.ArgumentParser(description="")

    # #########################################################################
    # ######         INPUT AND OUTPUT PATHS AND DIRS         ##################
    # #########################################################################

    ##################
    # list_of_tuples_of_pairs_of_bootstrap_paths
    ##################
    help_msg = "list_of_tuples_of_pairs_of_bootstrap_paths"
    default = config.list_of_tuples_of_pairs_of_bootstrap_paths
    parser.add_argument("list_of_tuples_of_pairs_of_bootstrap_paths", nargs="+", help=help_msg, default=default, type=str,)
    
    
    ##################
    # path_lcs_eval_statistical_tests
    ##################
    help_msg = "path_lcs_eval_statistical_tests"
    default = config.path_lcs_eval_statistical_tests
    parser.add_argument("path_lcs_eval_statistical_tests", help=help_msg, default=default, type=str)

   

    return parser.parse_args(args)


# #########################################################################
# #########################################################################
# ######          MAIN PROGRAM               ##############################
# #########################################################################
# #########################################################################

if __name__ == "__main__":

    logger.info("Start Statistical Tests")
    local_run = True  # Put False if you want to run the script from Command line
    if local_run:
        #######  PARAMETERS - ARGS   ##############################
        list_of_tuples_of_pairs_of_bootstrap_paths = config.list_of_tuples_of_pairs_of_bootstrap_paths
        expdir = config.path_lcs_eval_statistical_tests
       
    else:
        args = parse_args(sys.argv[1:])
        ######  INPUT AND OUTPUT PATHS AND DIRS  ##################
        list_of_tuples_of_pairs_of_bootstrap_paths = args.evaluate_publication_series
        expdir = args.path_lcs_eval_statistical_tests
        

    statistical_test_main(list_of_tuples_of_pairs_of_bootstrap_paths,expdir,)
