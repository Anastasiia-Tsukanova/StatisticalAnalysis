#!/usr/bin/env python

# conda activate normalityindependence

from scipy.stats import shapiro, normaltest, anderson, norm, gaussian_kde, ks_2samp, mannwhitneyu, ttest_ind
import scipy.stats
from os import walk, makedirs
from os.path import basename as bname
from os.path import join, splitext, exists
import pandas as pd
import numpy as np
from math import sqrt

data_dir = 'data/'      # "/home/anastasiia/Data/"
files = ["abc2018.csv", "abc2019.csv"]          # Don't use "4dcollab.csv"
roles = ["Archi", "Enge", "Civil", "Pedago", "Constr", "Client", "ALL"]

result_dir = 'res/'     # '/home/anastasiia/Data/result_dir_2018_2019'

def ensure_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)

def ensure_dirs(*args):
    dir_list = args
    for d in dir_list:
        ensure_dir(d)

def check_normality(vals, output_f):
    with open(output_f, 'w') as norm_f:
        norm_report = ['Sample size: %d' % vals.size]
        if vals.size < 8:
            norm_f.write('\n'.join(norm_report))
            return
        norm_report += ['Shapiro-Wilk']
        stat, p = shapiro(vals)
        norm_report += ['Statistics=%.3f, p=%.3f' % (stat, p)]
        # interpret
        alpha = 0.05
        if p > alpha:
            norm_report.extend(['Sample looks Gaussian (fail to reject H0)'])
        else:
            norm_report.extend(['Sample does not look Gaussian (reject H0)'])
        norm_report.extend(['-----'])
        norm_report.extend(['D\'Agostino\'s K^2 Test'])
        stat, p = normaltest(vals)
        norm_report.extend(['Statistics=%.3f, p=%.3f' % (stat, p)])
        # interpret
        alpha = 0.05
        if p > alpha:
            norm_report.extend(['Sample looks Gaussian (fail to reject H0)'])
        else:
            norm_report.extend(['Sample does not look Gaussian (reject H0)'])
        norm_report.extend(['-----'])
        norm_report.extend(['Anderson'])
        result = anderson(vals)
        norm_report.extend(['Statistic: %.3f' % result.statistic])
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                norm_report.extend(['%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv)])
            else:
                norm_report.extend(['%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv)])
        norm_f.write('\n'.join(norm_report))

def interprete_indep(ks_stat, ks_p, mw_stat, mw_p, tt_stat, tt_p, equal_var, what_we_are_comparing="", size1=None, size2=None, params=(0.05, 1.224)):
    p_val, ks_coef = params
    # ks_compare = ks_coef*sqrt(float(size1 + size2)/(size1*size2))
    # if ks_p < p_val and ks_stat > ks_compare:
    # 0: Not different, 1: different, 2: mixed results
    ks_res = 1 if ks_p < p_val else 0
    mw_res = 1 if mw_p < p_val else 0
    tt_res = 1 if tt_p < p_val else 0
    res = ks_res if ks_res == mw_res == tt_res else 2
    equal_var = "same" if equal_var == True else "different"
    if res == 0:
        print("%seeming to be same with %s variances..." % ("S" if what_we_are_comparing == "" else what_we_are_comparing + " s", equal_var))
    if res == 1:
        print("%seeming to be different with %s variances..." % ("S" if what_we_are_comparing == "" else what_we_are_comparing + " s", equal_var))
    if res == 2:
        print("Getting mixed results%s with %s variances..." % ("" if what_we_are_comparing == "" else " for " + what_we_are_comparing, equal_var))
    return res

def check_indep(data1, data2, output_file, alpha=0.05):
    ks_stat, ks_p = ks_2samp(data1, data2)
    mw_stat, mw_p = mannwhitneyu(data1, data2)
    equal_var = False if scipy.stats.f.cdf(np.var(data1)/np.var(data2), data1.size-1, data2.size-1) > alpha else True
    tt_stat, tt_p = ttest_ind(data1, data2, 0, equal_var)
    interpret = interprete_indep(ks_stat, ks_p, mw_stat, mw_p, tt_stat, tt_p, equal_var)
    with open(output_file, 'w') as indep_f:
        indep_f.write("Kolm,%f,%f\nMann,%f,%f\nTT,%s,%f,%f,\nS.sizes,%d,%d" % (ks_stat, ks_p, mw_stat, mw_p, 'sigma1=sigma2' if equal_var else 'sigma1!=sigma2', tt_stat, tt_p, data1.size, data2.size))
    return interpret


data_by_role = dict()
data_by_role_by_year = dict()
for role in roles:
    data_by_role[role] = list()
    for f in files:
        data_by_role_by_year[f[:f.find('.')]+'_'+role] = list()
for f in files:
    df = pd.read_csv(join(data_dir, f), usecols=["Role", "Score"])
    for index, entry in df.iterrows():
        entry_role, entry_sum = entry["Role"], entry["Score"]
        data_by_role[entry_role].append(entry_sum)
        data_by_role_by_year[f[:f.find('.')]+'_'+entry_role].append(entry_sum)
        data_by_role["ALL"].append(entry_sum)
        data_by_role_by_year[f[:f.find('.')]+'_ALL'].append(entry_sum)

norm_dir = join(result_dir, 'normality')
indep_dir = join(result_dir, 'independence_prec_vals')
ensure_dirs(result_dir, norm_dir, indep_dir)
list_to_check = list()
for role in roles:
    if data_by_role[role]:
        data_by_role[role] = np.array(data_by_role[role])
    for f in files:
        if data_by_role_by_year[f[:f.find('.')]+'_'+role]:
            data_by_role_by_year[f[:f.find('.')]+'_'+role] = np.array(data_by_role_by_year[f[:f.find('.')]+'_'+role])
for key, vals in data_by_role.items():
    if vals != []:
        check_normality(vals, join(norm_dir, key+'_normality.txt'))
        
for key, vals in data_by_role_by_year.items():
    if vals != []:
        check_normality(vals, join(norm_dir, key+'_normality.txt'))

role_pairs = [role1 + '_vs_' + role2 for k, role1 in enumerate(roles) for role2 in roles[k+1:] if role1 != "ALL" and role2 != "ALL" and data_by_role[role1] != [] and data_by_role[role2] != []]
role_pairs_in_a_year = [role1 + '_vs_' + role2 + '_in_' + f[:f.find('.')] for f in files for k, role1 in enumerate(roles) for role2 in roles[k+1:] if role1 != "ALL" and role2 != "ALL" and data_by_role_by_year[f[:f.find('.')]+'_'+role1] != [] and data_by_role_by_year[f[:f.find('.')]+'_'+role2] != []]

independence = [[], [], []]
for key in role_pairs + role_pairs_in_a_year:
    print(key)
    indep_file = join(indep_dir, key+'_independence.txt')
    parts = key.split('_')
    role1, role2 = parts[0], parts[2]
    if len(parts) <= 3:
        data1, data2 = data_by_role[role1], data_by_role[role2]
    else:
        data1, data2 = data_by_role_by_year[parts[-1] + "_" + role1], data_by_role_by_year[parts[-1] + "_" + role2]
    interpret = check_indep(data1, data2, indep_file)
    independence[interpret].append(key)

interpretation_file_names = ['same-distr.txt', 'diff-distr.txt', 'mixed-res.txt']
for interpet_f_id, interpret_f_name in enumerate(interpretation_file_names):
    with open(join(result_dir, interpret_f_name), 'w') as interpret_f:
        interpret_f.write('\n'.join(sorted(independence[interpet_f_id])))