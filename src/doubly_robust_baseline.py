import numpy as np
from doubleml import DoubleMLData
import pandas as pd
import time
from src.utils import rmse, log_results_evaluate
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from doubleml import DoubleMLPLR, DoubleMLData
from argparse import ArgumentParser

def run_double_ml_plr(data, dataframe):
    if data.startswith('lalonde_cps'):
        learner_m = DecisionTreeClassifier(max_depth=3)
        learner_l = LinearSVR(C=10000)
        dml_data = DoubleMLData(dataframe,
                                y_col='y',
                                d_cols='t',
                                x_cols=['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74',
                                        're75'])
    elif data.startswith('lalonde_psid'):
        learner_m = DecisionTreeClassifier(max_depth=4)
        learner_l = LinearSVR(C=10000)
        dml_data = DoubleMLData(dataframe,
                                y_col='y',
                                d_cols='t',
                                x_cols=['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74',
                                        're75'])
    elif data.startswith('twins'):
        learner_m = LogisticRegression(C=100, solver='liblinear')
        learner_l = KernelRidge(alpha=0.0001)
        dml_data = DoubleMLData(dataframe,
                                y_col='y',
                                d_cols='t',
                                x_cols=['eclamp', 'gestatcat1', 'gestatcat2', 'gestatcat3', 'gestatcat4',
                                        'gestatcat5', 'gestatcat6', 'gestatcat7', 'gestatcat8', 'gestatcat9',
                                        'gestatcat10', 'gestatcat1.1', 'gestatcat2.1', 'gestatcat3.1', 'bord',
                                        'gestatcat4.1', 'gestatcat6.1', 'gestatcat7.1', 'gestatcat8.1',
                                        'gestatcat9.1', 'gestatcat10.1', 'gestatcat1.2', 'gestatcat2.2',
                                        'gestatcat3.2', 'gestatcat4.2', 'gestatcat5.1', 'gestatcat6.2',
                                        'gestatcat7.2', 'gestatcat8.2', 'gestatcat5.2', 'gestatcat9.2',
                                        'gestatcat10.2', 'othermr', 'dmar', 'csex', 'cardiac', 'uterine',
                                        'lung', 'diabetes', 'herpes', 'anemia', 'hydra', 'chyper', 'phyper',
                                        'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'hemo', 'tobacco',
                                        'alcohol', 'orfath', 'adequacy', 'drink5', 'mpre5', 'meduc6', 'mrace',
                                        'ormoth', 'frace', 'birattnd', 'stoccfipb_reg', 'mplbir_reg', 'cigar6',
                                        'mager8', 'pldel', 'brstate_reg', 'feduc6', 'dfageq', 'nprevistq',
                                        'data_year', 'crace', 'birmon', 'dtotord_min', 'dlivord_min'])
    else:
        raise ValueError("Invalid data argument. Expected data name to start with: 'lalonde_cps', 'lalonde_psid' or 'twins'")

    ml_l = clone(learner_l)
    ml_m = clone(learner_m)

    obj_dml_plr_bonus = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2)
    obj_dml_plr_bonus.fit()

    ATE_AIPW = obj_dml_plr_bonus.coef[0]

    return ATE_AIPW

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)

    args = parser.parse_args()

    # get true ATE: mean of y1 - mean of y0
    dataframe = pd.read_csv(args.data_file)
    ATE_true = dataframe['y1'].mean() - dataframe['y0'].mean()
    print("true ATE:", ATE_true)
    ATE_AIPW = run_double_ml_plr(args.data_name, dataframe)
    print(f"The coefficient of 't' is: {ATE_AIPW}")
    rmse_AIPW = rmse(ATE_AIPW, ATE_true)
    print(f"RMSE from DRML: {rmse_AIPW:.4f}")

    ATE_AIPW_baseline = ATE_AIPW

    # Gather results in a dictionary
    results_bl = {
        'ATE_AIPW_baseline': ATE_AIPW_baseline
    }

    # Log the results
    log_results_evaluate(results_bl, config=None, results_file=args.results)





