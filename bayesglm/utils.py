import pandas
import logging

from typing import Dict, List
from distributions import Distribution, Uninformed
from itertools import product

logger = logging.getLogger(__name__)


def resolve_priors(priors: Dict[str, Distribution]) -> None:

    for prior_name, dist in priors.items():
        if not isinstance(dist, Distribution) or not isinstance(prior_name, str):
            raise RuntimeError("Prior dictionary must have string keys and Distribution object values.")
        
        for arg in dist.ARG_VALUES:
            if isinstance(arg, str):
                if arg not in priors:
                    raise RuntimeError("Could not find '%s' in prior dict." % arg)
            elif not isinstance(arg, (int, float)):
                raise RuntimeError("Distribution parameters should be numeric or a string pointing to other distribution.")



def prior_deps_recurse(prior_name, prior_map, rec_depth=0):
    warn = []
    if rec_depth > 100:
        raise RuntimeError("High recursion depth, may be a circular prior dependency.")

    deps = []
    prior_dist = prior_map.get(prior_name, None)
    
    if not prior_dist:
        warn.append(prior_name)
        prior_dist = Uninformed()
        prior_map.update({prior_name : prior_dist})
    for arg in prior_dist.ARG_VALUES:
        if isinstance(arg, str):
            deps.append(arg)
            deps.extend(prior_deps_recurse(arg, prior_map, rec_depth+1))

    #if warn:
    #    logger.warning("Following parameters had no prior specified, imputed 'Uninformed()' : %s" % warn)

    return deps


def remove_and_sort_priors(prior_map, family_args, beta_terms):

    used_priors = set()
    lhs = []
    rhs = []
    
    for arg in family_args:
        if isinstance(arg, str):
            lhs.append(arg)
            used_priors.add(arg)
            deps = prior_deps_recurse(arg, prior_map)
            lhs.extend(deps)
            used_priors.update(deps)
    
    for arg in beta_terms:
        if isinstance(arg, str):
            rhs.append(arg)
            used_priors.add(arg)
            deps = prior_deps_recurse(arg, prior_map)
            lhs.extend(deps)
            used_priors.update(deps)

    unused_priors = set(prior_map) - used_priors
    if unused_priors:
        logger.info("Removed the following priors as they are not in the model: %s" % str(unused_priors))
    
    return sorted(set(lhs))+ sorted(rhs)

def get_prior_specified_and_params(prior_list, prior_map):

    specified = []
    params = []

    for prior in prior_list:
        prior_dist = prior_map.get(prior)
        spec = []
        args = []
        for arg in prior_dist.ARG_VALUES:
            if isinstance(arg, str):
                spec.append(False)
                args.append(prior_list.index(arg))
            else:
                spec.append(True)
                args.append(arg)
        
        specified.append(spec)
        params.append(args)
    
    return specified, params

def check_formula(formula: str):
    if not isinstance(formula, str):
        raise TypeError("formula should be of type 'str', got %s." % str(type(formula)))
            


def check_distribution(dist):
    pass



def get_terms(formula: str, data_columns: List[str]):
    #first condense formula
    formula = formula.replace(" ", "")
    to_add = set()
    to_remove = set()
    response = None
    to_split = []

    #now do response split 
    res_split = formula.split("~")
    response = res_split[0]

    rhs = res_split[1]

    if "." in rhs:
        to_add.update(data_columns)
        to_add.remove(response)
    
    #now do add split
    add_split = rhs.split("+")
    for term in add_split:
        if "-" not in term:
            to_add.add(term)
        else:
            to_split.append(term)

    for term in to_split:
        sub_split = term.split("-")
        to_add.add(sub_split[0])
        to_remove.update(sub_split[1:])

    if "." in to_add:
        to_add.remove(".")
    
    if "1" not in to_remove:
        to_add.add("Intercept")
        if "1" in to_add:
            to_add.remove("1")
    else:
        to_remove.remove("1")
        to_remove.add("Intercept")
    
    return response, to_add, to_remove


def split_interaction_terms(to_add: set, to_remove: set):

    to_update = set()
    to_subtract = set()

    for term in to_add:
        if "*" in term:
            to_subtract.add(term)
            to_update.add(tuple(sorted(term.split("*"))))

    to_add = to_add.union(to_update) - to_subtract

    to_update = set()
    to_subtract = set()
    for term in to_remove:
        if "*" in term:
            to_subtract.add(term)
            to_update.add(tuple(sorted(term.split("*"))))
        
    to_remove = to_remove.union(to_update) - to_subtract

    return to_add, to_remove


def encode_discrete_cols(col_set: set, data: pandas.DataFrame):

    discrete_cols = [col for col in data.columns if data.dtypes[col] == "object"]
   
    used_cols = set()

    for col in discrete_cols:

        terms_to_remove = set()
        terms_to_add = set()

        for term in col_set:
            if col == term:
                components = tuple(col+comp for comp in data[col].unique()[1:]) #not including intercept absorbed term
                terms_to_remove.add(term)
                terms_to_add.update(components)
                used_cols.add(col)
            elif isinstance(term, tuple) and col in term:
                components = tuple(col+comp for comp in data[col].unique()[1:]) #not including intercept absorbed term
                temp = list(term)
                temp.remove(col)
                temp = tuple(temp)
                terms_to_remove.add(term)
                terms_to_add.update(list(product(temp, components)))
                used_cols.add(col)
        
        col_set = col_set.union(terms_to_add)
        col_set = col_set - terms_to_remove

    return col_set, used_cols

def encode_data_cols(cols: List[str], data: pandas.DataFrame):
    copy = data.copy()
    for col in cols:
        components = tuple(copy[col].unique()[1:])
        for component in components:
            new_col = "%s%s" % (col, component)
            copy[new_col] = (copy[col] == component).astype(int)
        copy = copy.drop(col, axis=1)
    
    return copy

def construct_model_mat(to_add: set, to_remove: set, data: pandas.DataFrame):
    out = pandas.DataFrame(index=data.index)

    to_add = sorted([i for i in to_add if isinstance(i, tuple)]) + \
             sorted([i for i in to_add if isinstance(i, str)]) 
    to_remove = sorted([i for i in to_remove if isinstance(i, tuple)]) + \
                sorted([i for i in to_remove if isinstance(i, str)])

    for term in to_add:
        if isinstance(term, str):
            if term == "Intercept":
                out[term]=1
            else:
                out[term] = data[term]
        elif isinstance(term, tuple):
            name = "*".join(term)
            out[name] = 1
            for col in term:
                out[name] *= data[col]
    
    for term in to_remove:
        if term in out.columns:
            out=out.drop(term, axis=1)
    
    return out

def model_matrix(formula: str, data: pandas.DataFrame):
    
    response, to_add, to_remove = get_terms(formula, data.columns)
    response_vec = data[response]
    to_add, to_remove = split_interaction_terms(to_add, to_remove)

    #begin constructing data frame
    to_add, used_add = encode_discrete_cols(to_add, data)
    to_remove, used_remove = encode_discrete_cols(to_remove, data)

    data = encode_data_cols(used_add.union(used_remove), data)

    out = construct_model_mat(to_add, to_remove, data)
    
    return response_vec, out





