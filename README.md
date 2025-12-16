# CADe/CADx Performance Evaluation Tutorial

This repository provides a complete evaluation of the performance of CADe (Computer-Aided Detection) and CADx (Computer-Aided Diagnosis) systems. 
It only adresses single "target" detection system, nambely binary detection and classification systems (e.g malignant nodule), and computes ROCs, Precison-Recall, FROCs (with severall Operating Points spacifictions), bootstrapps, Confidence Intervals, statistical comparison tests, across sets of detection predictions (...). Simply, it provides a standard and generic CADe/CADx evaluation process. 

It deserves two purposes that drives 2 independent usage: first to reproduce all the results of the preprint "Rethinking Lung Cancer Screening: AI Nodule Detection and Diagnosis Outperforms Radiologists, Leading Models, and Standards Beyond Size and Growth" ([https://arxiv.org/abs/2512.00281](https://arxiv.org/abs/2512.00281)), second to provide evaluation process system for other CADe/CADx.

## 1. Install

### 1.1 UV - Ultra-fast Python package manager & Git

This repository uses UV to manage the dependencies, environment, python version (...) of the repository, and wenever you would not have it yet, you need to install it first for a direct, fast and easy usage.
UV is an ultra-fast Python package manager (see complete on [complete UV installation guidelines](https://uv.pypa.io/en/stable/installation/)).
To install UV, run the following command in your terminal depending on wheither your system is Linux-based or Windows:

Linux / macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or if you don't have curl:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
irm (Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1).Content | iex
```

Whenever you would not have it yet, install Git ([complete GIT installation guidelines](https://git-scm.com/book/fr/v2/D%C3%A9marrage-rapide-Installation-de-Git)): 
Linux / macOS:
```bash
sudo apt-get install git
```
For windows, download the [installer](https://git-scm.com/download/win) or use the package [Chocolatey Git](https://chocolatey.org/packages/git) for automatic set up.

### 1.2 Clone the repositopry 

Create the directory where to save the repository on your hard disc and in your terminal go at this repository (called here "CASe-CADx-evaluation"), and run the command:
```bash
git clone https://github.com/EYONIS-AIDS-DS/CADe-CADx-evaluation.git
```
or use some SSH key, or dowload the zip from "code" in Github and unzip it in your directory.

### 1.3 Install dependencies

To create the virtual environment:
```bash
uv venv
```
To install the required dependencies, run the following command in your terminal at:
```bash
uv sync
```
To activate the virtual environment:

```bash
source .venv/bin/activate
```
### 1.4 Data Files Setup

This repository uses Git LFS (Large File Storage) for large CSV files.
 
**Before running the evaluation scripts, you must Install Git LFS:**
 
   ```bash
   # macOS
   brew install git-lfs
   # Windows
   choco install git-lfs
   # OR download from https://git-lfs.github.com/
   # Linux (Ubuntu/Debian)
   sudo apt-get install git-lfs
   ```
Then, pull the data files:
```bash
git lfs install
git lfs pull
```
Verify the files are correct:
```bash
head -1 data/data_lesions/lesions.csv
```
If you experience issues with Git LFS, you can manually download the files:
```bash
$files = @(
    "lesions.csv",
    "lesions_radiologists.csv",
    "lesions_longitudinal.csv",
    "lesions_model_CADe.csv",
    "lesions_nndetection_baumgartner_CADe.csv",
    "lesions_nndetection_CADe.csv",
    "lesions_nndetection_CADx.csv"
)
$base_url = "https://media.githubusercontent.com/media/EYONIS-AIDS-DS/CADe-CADx-evaluation/refs/heads/main/data/data_lesions%22
foreach ($file in $files) {
    Invoke-WebRequest -Uri "$base_url/$file" -OutFile "data/data_lesions/$file"
} 
```

## 2. Structure of the repository

The stucture of the reposity is the following (including the repositories of the output processing that are not commited): 

```
CADe_CADx_evaluation
│   README.md
│   config_paper.py  
│   run_paper_evaluation.py   
│
└───Data
│   │CADe_CADx_evaluate.log  (OUTPUT)
│   │
│   └───data_series          (INPUT)
│   │   │series.csv          
│   │  
│   └───data_lesions         (INPUT)
│   │   │lesions.csv         
│   │   │... 
│   │     
│   └───evaluate_series      (OUTPUT)
│   │   └─── figure_1  
│   │   │     └───4_radiologist_prediction_test3     
│   │   │         │roc_curve_with_op_test3.png  
│   │   │         │... 
│   │   │    
│   │   └─── ...        
│   │     
│   └───evaluate_lesions     (OUTPUT)
│   │   └─── figure_2  
│   │   │     └───4_radiologist_prediction_test3     
│   │   │         │froc_curve_with_2_op_test3.png  
│   │   │         │... 
│   │   │    
│   │   └─── ...        
│   │     
│   └───statistical_tests     (OUTPUT)
│       └─── figure_1  
│       │    │test_results_AUC_model_vs_4radiolog_test3.csv  
│       │    │... 
│       │    
│       └─── ...                
│   
└───evaluate_common            (CODE)
│   │roc.py   
│   │precision_recall.py
│   │sample_size_analysis.py
│   │roc_confidence_interval.py
│   │plot_score_distribution_benign_cancer.py
│   │logger.py
│   │sens_spec.py
│  
└───evaluate_lesions           (CODE)
│   │evaluate_lesions.py   
│   │froc.py
│   │plot_diameter_prediction_distributionss.py
│  
└───evaluate_series            (CODE)
│   │evaluate_series.py   
│  
└───statistical_tests          (CODE)
│   │statistical_tests.py   

```

### 2.1 Configuration
All the parameters of the package are defined in "config_paper.py". It includes input and output data paths, name of the predictions, names of the subsets sample of an evaluation, names of the label GT, percentage of the confidence intervals, number of bootstrap, a fast computation option, operating points thresholds and labels. It also defines lists of evalutations specifying predictions, labels, subsets (...) to be runed grouped by figures. To apply the package to new models and dataset, you may either rewrite this configuration and define your own lists of evaluations (complex case of multiple evaluations), or directly run the submodules functions (detailed bellow). By default, this configuration script reproduces all the figures, results of the paper arXiv (to complete).

### 2.2 Input
The input are stored in .\data\data_series and .\data\data_lesions directory, for series/patient level and nodules/lesions level input respectively.
They are csv files (e.g. series.csv and lesions.csv). Each row is 1 sample, each column is a feature of the sample. There are 4 kind of features used by the evaluation:

* predictions: they are numeric (float or int) commonly a probability prediction of a model for the sample, but can be also a (measure) psychophysical assement of a human (e.g. an expert radiologist), or a numerical variable associated with the sample (eg. size) that may have a predictive value of the binary detection-classification.
* labels: this are the ground truth associated with the sample. They are binary: either (0,1) or Boolean (true or 1 indicate the postive class, False or 0 indicate the negative class)
* identifiers: commonly in medical imaging, patient_id, series_uid, Time_point
* features-variables: they are varaibles associated to the sample, on which you may stratify your evalutation (e.g. age, gender, slice-thickness, manufacturer, spiculation...)
* test set name: these are boolean variable (indicator functions) that indicate wheither the sample pertain or not to a subset (this has the same role as stratification)
In addition for the lesion/nodule level only, there is a required column "detection_status" that can take either "TP","FP","FN" (for "True Positive", "False Positive", and "False Negative"...) values as a result the output of the pairing of the detection with the GT (see bellow). Note that a "False negative" detections are attributed a prediction 0 (or least score) by the classification evaluation algorithm.  
 
### 2.3 modules "evaluate_lesions" and "evaluate_series"
The submodules "evaluate_lesions" and "evaluate_series" make essentially the same tasks and computation but either at series/patient level or at nodule/lesion level.
Those 2 levels are dissociated because patient/series level performance evaluation is straighforward from the GT input (e.g. cancer or not), whereas nodule/lesion level performance requires a prior pairing of the detections with the lesions in the GT.
This pairing is not furnished (yet) with the repository, we used standard (for 3D pairing) IoU based pairing with 0.1 threshold as furnished and recommended by [Jaeger et al.](https://arxiv.org/abs/1811.08661).
Moreover, lesion/nodule level is also commonly evaluated using FROCs which does not make sens at scan/patient level. Both modules call functions like roc.py (etc..) in the "evaluate_common" library.
They compute ROCs, FROCs (at lesion levels), etc. (see bellow in the output descriptions).

### 2.4 module "statistical_tests"
The "statistical_tests" module is based on Bootstrap methods conceived by [Efron and Tibshirani](https://www.hms.harvard.edu/bss/neuro/bornlab/nb204/statistics/bootstrap.pdf). The bootstrap samples are computed with replacement.  It is recommended because Bootstrap methods are non-parametric (and does not make assumption on the distribution that should be verified, or use parameter that should be fitted), and can be applied generically using a common framework to wide range of observable (on AUC ROC and sensitivity of a given Operating Point of the ROC similarly). It takes a list of paths of pairs of npy saved vectors (in evaluate lesions or series) of n bootstrap samples metric values (AUC, accuracy, sensitivity, specificity....) and run all the list of tests. It implements each time 2 statistical tests using [scipy stats](https://docs.scipy.org/doc/scipy/reference/stats.html): 
* a superiority t-test with unequal variance (one-sided Welch t-test, 1 st prediction vs. 2nd)
* a superiority test assuming equal variance (t-test, 1 st prediction vs. 2nd) 
It  return a csv with the result of all pairs of tests (p value, acceptance (strong, very strong, moderate, rejected)), but also a figure of the 2 bootstrapps distributions. 
This module can only be runned after "evaluate_lesions" and-or "evaluate_series" as it depends on their Bootstraps vector .npy file outputs (they are its only inputs). 

### 2.5 Outputs
The input are stored in .\data\evaluate_series and .\data\evaluate_lesions directory, for series/patient level and nodules/lesions level input respectively.
They are csv files, figures both saved in .png and .svg (allowing to further edit them in vectorial format), numpy arrays of bootstrapps samples (for statistical test). 
The saved figures are:
* ROCs, with Operating Points (by default the Youden Index Maximum OP is given, but a list of OP threshold and labels can be given) 
* Precision Recall curve 
* the distribution of predictions for each labels
* FROCs  (only at lesion level): with operating points at the point closest to 0.5FP/scan and 1FP/scan 

The saved csv are:
* sample sizes: with total sample size, imbalance, and sample size in each class.
* "roc_CI_bootstrap_hanley" csv: with AUC ROC confidence intervals (lower and upper) obtained using bootstrap methods and using [Hanley & McNeil method](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed). 
* "operating_point_performances" csv: with, for all operating points, the sens, spec, accuracy, their mean, lower and upper CI over n bootstrapps samples.  
* "operating_point_FROC_scores_at_0.5_and_1_FP_per_scan"  (only at lesion level): with, for operating points at the point closest to 0.5FP/scan and 1FP/scan , the sens and exact FP/scan, their mean, lower and upper CI over n bootstrapps samples.
* "test_results" csv (in statistical test folders):  with the result of all pairs of tests (p value, acceptance (strong, very strong, moderate, rejected)), but also a figure of the 2 bootstrapps distributions.

The saved numpy (.npy) files are the n_boostrapped vectors of AUC, sens, spec and accuracy.

## 3. Reproduce all papers figures, performances and tests in a single command from data inputs

The `run_paper_evaluation.py` script is designed to produce all the evaluation performance analyses presented in the paper (give arXiv link). This script integrates all the evaluation modules of package and generates the required outputs of each figures of the paper in the associated directories.
At series-patient and at nodule level, it generates all the output listed above for all figures of the paper, and all the statistical tests of the paper.
All its parameters are strored in `config_paper.py`, notably the path where to get the input csv data for patients and nodules predictions and the path where to store the output, along with the list of figures, stats and tests to produce. This config is specific to the paper, just rewrite another config keeping the same structure for another set of evaluations.

To make the whole paper evaluation, just run the following command in terminal: 
```
\CADe_CADx_evaluation>python3 run_paper_evaluation.py --fast_computation=true
```

or alternatively in a python script, write:

```
from  CADe_CADx_evaluation.run_paper_evaluation import paper_evaluation
fast_computation = True  # Set to True for faster computation during testing/debugging (nb bootstrap samples reduced to 50 and fast FROC computation)
paper_evaluation(fast_computation = fast_computation) 
```
In this demo example we settled the parameter "fast_computation" to True:  it allows a quite fast runing of the evaluation process (notably the number of bootstrapps samples is 50).

Although qualitatively the results does not differ much, if you want to reproduce the paper evaluation, you have to set the parameter "fast_computation" to False (and expect about a day of computation):

```
\CADe_CADx_evaluation> python3 run_paper_evaluation.py --fast_computation=false
```

or alternatively in a python script, write:

```
from  CADe_CADx_evaluation.run_paper_evaluation import paper_evaluation
fast_computation = False  # Set to True for faster computation during testing/debugging (nb bootstrap samples reduced to 50 and fast FROC computation)
paper_evaluation(fast_computation = fast_computation) 

```

## 4. Submodules "evaluate_series", "evaluate_lesions", and "statistical_tests" independent usage

We can now investigate the uses of the different submodules independently. To do so, you can either use the input csv provided in \data\data series or create a new one from scratch:

```
import CADe_CADx_evaluation.config_paper as config
import pandas as pd
from CADe_CADx_evaluation.evaluate_series import evaluate_series as eval_series
import numpy as np

path_data_series = config.path_data / "data_series" / "series_to_use.csv"
sample_size = 1000
np.random.seed(42) 
data = {
    "series_uid": np.arange(1, sample_size + 1),  # Unique identifiers from 1 to 1000
    "prediction_to_use": np.random.rand(sample_size),  # Random float values between 0 and 1
    "label_to_use": np.random.randint(0, 2, size=sample_size),  # Random binary values (0 or 1)
    "subset_to_use": np.random.choice(["TRUE", "FALSE"], size=sample_size, p=[0.9, 0.1])  # 90% TRUE, 10% FALSE
}
df = pd.DataFrame(data)
df["prediction_to_use"]= ((df["label_to_use"]*2)-1) * np.random.rand(sample_size) *0.5 +0.5  # make predictions correlated with labels
df.to_csv(path_data_series, index=False)
```

### 4.1 Submodule "evaluate_series"

We can run the evaluation of series prediction we just created, by defining the arguments values and running:
```
from CADe_CADx_evaluation.evaluate_series import evaluate_series as eval_series
path_data_series = config.path_data / "data_series" / "series_to_use.csv"
path_model_evaluate_series = config.path_model_eval / "evaluate_series" / "figure_test"
set_name = "subset_to_use"
prediction = "prediction_to_use"
label_name = "label_to_use"
operating_point_thresholds = [0,]
operating_point_labels = ["Youden Index Max"]
nb_bootstrap_samples = 500
confidence_threshold = 0.95
eval_series.evaluate_serie_main(path_data_series ,  # path_to_load_csv_serie,
                                path_model_evaluate_series ,  # expdir,
                                set_name,  # set_name,
                                prediction,  # prediction,
                                label_name,  # label_name,
                                operating_point_thresholds,  # operating_point_thresholds,
                                operating_point_labels,
                                nb_bootstrap_samples,  # nb_bootstrap_samples,
                                confidence_threshold,)  # confidence_threshold,config.confidence_threshold,  # confidence_threshold,
```

### 4.2 Submodule "evaluate_lesions"

### 4.3 Submodule "statistical_tests"
