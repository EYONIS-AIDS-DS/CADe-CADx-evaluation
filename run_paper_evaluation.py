
import timeit
import config_paper as config
from CADe_CADx_evaluation.evaluate_common.logger import logger
import argparse
import sys





def paper_evaluation(
    fast_computation = config.fast_computation,
    nb_bootstrap_samples = config.nb_bootstrap_samples,
    confidence_threshold = config.confidence_threshold,    
    path_model_eval = config.path_model_eval,
    run_evaluate_series = config.run_evaluate_series,
    path_model_evaluate_series = config.path_model_evaluate_series,
    list_series_evaluations_figure = config.list_series_evaluations_figure,
    run_evaluate_lesions = config.run_evaluate_lesions,
    path_model_evaluate_lesions = config.path_model_evaluate_lesions,
    list_lesions_evaluations_figure = config.list_lesions_evaluations_figure,
    run_statistical_tests = config.run_statistical_tests,
    path_model_statistical_tests = config.path_model_statistical_tests,
    list_statistical_tests_figure = config.list_statistical_tests_figure,   
)-> None:


    path_model_eval.mkdir(parents=True, exist_ok=True)
    if fast_computation:
        nb_bootstrap_samples = 50
    ###################################################################################################
    ##############################         EVALUATION  SERIES          ################################
    ################################################################################################### 

    if run_evaluate_series:

        logger.info("Running evaluate series")
        from CADe_CADx_evaluation.evaluate_series import evaluate_series as eval_series
        path_model_evaluate_series.mkdir(parents=True, exist_ok=True)
        for (list_series_evaluations, figure_name, path_data_series) in list_series_evaluations_figure:
            start = timeit.default_timer()
            for (set_name, prediction, label_name, operating_point_thresholds, operating_point_labels) in list_series_evaluations:
                logger.info(f"###############################    {figure_name}    ################################################")
                logger.info(f"Evaluating series for set: {set_name}, prediction: {prediction}, label_name: {label_name}")
                eval_series.evaluate_serie_main(path_data_series ,  # path_to_load_csv_serie,
                                            path_model_evaluate_series / figure_name,  # expdir,
                                            set_name,  # set_name,
                                            prediction,  # prediction,
                                            label_name,  # label_name,
                                            operating_point_thresholds,  # operating_point_thresholds,
                                            operating_point_labels,
                                            nb_bootstrap_samples,  # nb_bootstrap_samples,
                                            confidence_threshold,)  # confidence_threshold,config.confidence_threshold,  # confidence_threshold,  
            logger.info(f"The time to evaluate series {figure_name} is: {timeit.default_timer() - start}")


    ###################################################################################################
    ##############################         EVALUATION  LESIONS         ################################
    ################################################################################################### 

    if run_evaluate_lesions:
        logger.info("Running evaluate lesions")
        from CADe_CADx_evaluation.evaluate_lesions import evaluate_lesions as eval_lesion
        path_model_evaluate_lesions.mkdir(parents=True, exist_ok=True)
        for (list_lesions_evaluations, figure_name, path_data_lesions) in list_lesions_evaluations_figure:
            start = timeit.default_timer()
            for (set_name, prediction, label_name, operating_point_thresholds, operating_point_labels) in list_lesions_evaluations:
                logger.info(f"###############################    {figure_name}    ################################################")
                logger.info(f"Evaluating series for set: {set_name}, prediction: {prediction}, label_name: {label_name}")
                eval_lesion.evaluate_lesions_main(fast_computation,
                                            path_data_lesions ,  # path_to_load_csv_serie,
                                            path_model_evaluate_lesions / figure_name,  # expdir,
                                            set_name,  # set_name,
                                            prediction,  # prediction,
                                            label_name,  # label_name,
                                            operating_point_thresholds,  # operating_point_thresholds,
                                            operating_point_labels,
                                            nb_bootstrap_samples,  # nb_bootstrap_samples,
                                            confidence_threshold,
                                            fast_computation)  # confidence_threshold,config.confidence_threshold,  # confidence_threshold,  
            logger.info(f"The time to evaluate series {figure_name} is: {timeit.default_timer() - start}")
        

    ###################################################################################################
    ##############################         STATISTICAL TESTS           ################################
    ################################################################################################### 

    if run_statistical_tests:
        logger.info("Running statistical tests")
        #from lcseval.statistical_tests.statistical_tests import statistical_test_main as stat_test_main
        from CADe_CADx_evaluation.statistical_tests.statistical_tests import statistical_test_main as stat_test_main
        path_model_statistical_tests.mkdir(parents=True, exist_ok=True)
        for (list_of_tuples_of_pairs_of_bootstrap_paths, figure_name) in list_statistical_tests_figure:
            logger.info(f"###############################    {figure_name}    ################################################")
            path_model_eval_statistical_tests = path_model_statistical_tests / figure_name
            path_model_eval_statistical_tests.mkdir(parents=True, exist_ok=True)
            start = timeit.default_timer()
            stat_test_main(list_of_tuples_of_pairs_of_bootstrap_paths, path_model_eval_statistical_tests)
            logger.info(f"The time to evaluate series {figure_name} is: {timeit.default_timer() - start}")

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
    # fast_computation
    ##################
    help_msg = "fast_computation"
    default = config.fast_computation
    parser.add_argument(
        "--fast_computation",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,  # Default value is True
        help="Enable or disable fast computation (true/false). Default is true.",
    )

    ##################
    # nb_bootstap_samples
    ##################
    help_msg = "nb_bootstrap_samples"
    default = config.nb_bootstrap_samples
    parser.add_argument("--nb_bootstrap_samples", required=False, type=int, help=help_msg, default=default)

    ##################
    # confidence_threshold
    ##################
    help_msg = "confidence_threshold"
    default = config.confidence_threshold
    parser.add_argument("--confidence_threshold", required=False, type=int, help=help_msg, default=default)

    ##################
    # path_model_eval
    ##################
    help_msg = "path_model_eval"
    default = config.path_model_eval
    parser.add_argument("--path_model_eval", required=False, type=str, help=help_msg, default=default)
   
    ##################
    # run_evaluate_series
    ##################
    help_msg = "run_evaluate_series "
    default = config.run_evaluate_series
    parser.add_argument("--run_evaluate_series", required=False, type=bool, help=help_msg, default=default)

    ##################
    # path_model_evaluate_series
    ##################
    help_msg = "path_model_evaluate_series"
    default = config.path_model_evaluate_series
    parser.add_argument("--path_model_evaluate_series", required=False, type=str, help=help_msg, default=default)

    ##################
    # list_series_evaluations_figure
    ##################
    help_msg = "list_series_evaluations_figure"
    default = config.list_series_evaluations_figure
    parser.add_argument("--list_series_evaluations_figure", required=False, type=list, help=help_msg, default=default)

     ##################
    # run_evaluate_lesions
    ##################
    help_msg = "run_evaluate_lesions "
    default = config.run_evaluate_lesions
    parser.add_argument("--run_evaluate_lesions", required=False, type=bool, help=help_msg, default=default)

    ##################
    # path_model_evaluate_lesions
    ##################
    help_msg = "path_model_evaluate_lesions"
    default = config.path_model_evaluate_lesions
    parser.add_argument("--path_model_evaluate_lesions", required=False, type=str, help=help_msg, default=default)

    ##################
    # list_lesions_evaluations_figure
    ##################
    help_msg = "list_lesions_evaluations_figure"
    default = config.list_lesions_evaluations_figure
    parser.add_argument("--list_lesions_evaluations_figure", required=False, type=list, help=help_msg, default=default)

     ##################
    # run_statistical_tests
    ##################
    help_msg = "run_statistical_tests "
    default = config.run_statistical_tests
    parser.add_argument("--run_statistical_tests", required=False, type=bool, help=help_msg, default=default)

    ##################
    # path_model_statistical_tests
    ##################
    help_msg = "path_model_statistical_tests"
    default = config.path_model_statistical_tests
    parser.add_argument("--path_model_statistical_tests", required=False, type=str, help=help_msg, default=default)

    ##################
    # list_statistical_tests_figure
    ##################
    help_msg = "list_statistical_tests_figure"
    default = config.list_statistical_tests_figure
    parser.add_argument("--list_statistical_tests_figure", required=False, type=list, help=help_msg, default=default)
    
    return parser.parse_args(args)


# #########################################################################
# #########################################################################
# ######          MAIN PROGRAM               ##############################
# #########################################################################
# #########################################################################

if __name__ == "__main__":

    logger.info("Start PAPER EVALUATION")
    local_run = False  # Put False if you want to run the script from Command line else it will use the config parameters
    if local_run:
        #######  PARAMETERS - ARGS   ##############################
        fast_computation = config.fast_computation
        nb_bootstrap_samples = config.nb_bootstrap_samples
        confidence_threshold = config.confidence_threshold     
        path_model_eval = config.path_model_eval   
        run_evaluate_series = config.run_evaluate_series
        path_model_evaluate_series = config.path_model_evaluate_series  
        list_series_evaluations_figure = config.list_series_evaluations_figure
        run_evaluate_lesions = config.run_evaluate_lesions
        path_model_evaluate_lesions = config.path_model_evaluate_lesions
        list_lesions_evaluations_figure = config.list_lesions_evaluations_figure
        run_statistical_tests = config.run_statistical_tests
        path_model_statistical_tests = config.path_model_statistical_tests
        list_statistical_tests_figure = config.list_statistical_tests_figure
    
    else:
        args = parse_args(sys.argv[1:])
        ######  INPUT AND OUTPUT PATHS AND DIRS  ##################
        fast_computation = args.fast_computation
        nb_bootstrap_samples = args.nb_bootstrap_samples
        confidence_threshold = args.confidence_threshold
        path_model_eval =      args.path_model_eval
        run_evaluate_series = args.run_evaluate_series
        path_model_evaluate_series = args.path_model_evaluate_series
        list_series_evaluations_figure = args.list_series_evaluations_figure
        run_evaluate_lesions = args.run_evaluate_lesions
        path_model_evaluate_lesions = args.path_model_evaluate_lesions
        list_lesions_evaluations_figure = args.list_lesions_evaluations_figure
        run_statistical_tests = args.run_statistical_tests
        path_model_statistical_tests = args.path_model_statistical_tests
        list_statistical_tests_figure = args.list_statistical_tests_figure
        

    paper_evaluation(fast_computation,
    nb_bootstrap_samples,
    confidence_threshold,    
    path_model_eval,
    run_evaluate_series,
    path_model_evaluate_series,
    list_series_evaluations_figure,
    run_evaluate_lesions,
    path_model_evaluate_lesions,
    list_lesions_evaluations_figure,
    run_statistical_tests, 
    path_model_statistical_tests,
    list_statistical_tests_figure, 
    )