import numpy as np
from pymoo.factory import get_performance_indicator
import scipy.stats
import pandas as pd

def get_igd_history(problem, result):
    
    igd_function = get_performance_indicator("igd", problem.pareto_front(), norm_by_dist=True)
    
    igd_list = []
    
    for gen in result.history:
        
        F = gen.opt.get("F")
        igd = igd_function.do(F)
        igd_list.append(igd)
    
    return igd_list

def get_hv_history(problem, result):
    
    refpoint = problem.pareto_front().max(axis=0)
    nadir = refpoint
    ideal = problem.pareto_front().min(axis=0)
    hv_function = get_performance_indicator("hv", refpoint, zero_to_one=True,
                                            nadir=nadir, ideal=ideal)
    
    hv_list = []
    
    for gen in result.history:
        
        F = gen.opt.get("F")
        hv = hv_function.do(F)
        hv_list.append(hv)
    
    return hv_list

def get_experiment_metric(results, problem, metric):
    
    columns = np.arange(0, len(results), 1) + 1
    
    metrics = {}
    
    if metric == "igd":
        perf = get_igd_history
    elif metric == "hv":
        perf = get_hv_history
    else:
        raise KeyError("Unknown metric")
    
    for j, res in enumerate(results):
        
        metrics[columns[j]] = perf(problem, res)
    
    return metrics

class StatsSignificance:
    
    def __init__(self, lesser_is_better=True):
        
        self.lesser_is_better = lesser_is_better 
        self.best_median = None
        self.best_values = None
        self.best_label = None
        self.results = pd.DataFrame(columns=["median", "mean", "std",
                                             "shapiro", "levene", "t-test", "wilcoxon"])
    
    def get_best_from_dict(self, candidates):
        
        for key, value in candidates.items():
            
            if self.best_values is None:
                self.best_label = key
                self.best_values = value.iloc[-1, :]
                self.best_median = value.iloc[-1, :].median()
            else:
                if self.lesser_is_better and (value.iloc[-1, :].median() < self.best_median):
                    self.best_label = key
                    self.best_values = value.iloc[-1, :]
                    self.best_median = value.iloc[-1, :].median()
                elif (not self.lesser_is_better) and (value.iloc[-1, :].median() > self.best_median):
                    self.best_label = key
                    self.best_values = value.iloc[-1, :]
                    self.best_median = value.iloc[-1, :].median()
                else:
                    pass
    
    def compare(self, candidate, label=None):
        
        _, pval = scipy.stats.shapiro(candidate)
        shapiro = "* (%.3f)" % pval if pval >= 0.01 else ""

        _, pval = scipy.stats.levene(self.best_values, candidate)
        levene = "* (%.3f)" % pval if pval >= 0.05 else ""
        
        if self.lesser_is_better:
            alternative = "greater"
        else:
            alternative = "smaller"

        _, pval = scipy.stats.ttest_ind(candidate, self.best_values)
        ttest = "* (%.3f)" % pval if pval >= 0.05 else ""

        if len(self.best_values) == len(candidate):
            _, pval = scipy.stats.wilcoxon(candidate, self.best_values, zero_method="zsplit")
            wilcoxon = "* (%.3f)" % pval if pval >= 0.05 else ""
        else:
            wilcoxon = "x"
        
        if label is None:
            label = self.results.shape[0] + 1
        
        median_, mean_, std_ = candidate.median(), candidate.mean(), candidate.std()
        
        self.results.loc[label, :] = [median_, mean_, std_, shapiro, levene, ttest, wilcoxon]
