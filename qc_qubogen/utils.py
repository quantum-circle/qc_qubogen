import numpy as np
from sympy import Symbol, Poly, Add, Mul, Pow
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials


def sort_variables(var):
    """Returns sorted variables in an alphabetical order
    """
    variables_list = list(var)
    variables_name_list = [variable.name for variable in variables_list]
    max_str_length = max(set(map(len, variables_name_list)))
    for variable in variables_list:
        variable_name = variable.name
        variable.name = variable_name.zfill(max_str_length)
    sorted_variables = sorted(variables_list, key=lambda x: x.name)
    for variable in sorted_variables:
        variable.name = variable.name.lstrip('0')
    return sorted_variables


def special_constraint(constraint, kind):
    """special constraints are the following inequalities

    (1) x + y <= 1 | xy
    (2) x1 + x2 + x3 <= 1 | x1x2 + x1x3 + x2x3
    (3) x <= y | x - xy
    (4) x = y | x + y - 2xy
    (5) x + y >= 1 | 1- x- y + xy
    (6) x + y = 1 | 1 - x- y + 2xy

    Parameters
    ----------
    constraint : sympy.core.expr.Expr or subclasses
        constraint
    kind : string
        'lt', 'gt' or 'eq'

    Returns
    -------
    sympy.core.expr.Expr, bool
        constraint and Boolean, if it is a special constraint
    """
    constraint_as_dict = Poly(constraint).as_expr().as_coefficients_dict()
    constant = constraint_as_dict[1]
    if constant == 0:
        del constraint_as_dict[1]
        length = len(constraint_as_dict)
    else:
        length = len(constraint_as_dict)
        del constraint_as_dict[1]
    
    sufficient_cond = [all(
        coeff==1 or coeff==-1 for coeff in constraint_as_dict.values()),
                       constant in [0, -1],
                       length <= 4]
    if not all(sufficient_cond):
        return constraint, False
    
    condition_eq_1_2 = [any([length == 3, length == 4]),
                        kind == 'lt',
                        all(coeff==1 for coeff in constraint_as_dict.values()),
                        constant==-1]
    condition_eq_3_4 = [length == 2,
                        any([kind=='eq', kind=='lt']),
                        sum(constraint_as_dict.values())==0,
                        constant==0]
    condition_eq_5_6 = [length == 3,
                        any([kind=='eq', kind == 'gt']),
                        all(coeff==1 for coeff in constraint_as_dict.values()),
                        constant==-1]

    ### x + y <= 1, x + y + z <= 1
    if all(condition_eq_1_2):
        var_combinations = sorted(
            itermonomials(list(constraint.free_symbols), 2),
            key=monomial_key('lex', list(constraint.free_symbols))
        )
        new_constraint = sum([
            func for func in var_combinations
            if (not func.is_integer) and Poly(func).is_multivariate
        ])
        return new_constraint, True
    ### x <= y , x = y
    elif all(condition_eq_3_4):
        if kind == 'lt':
            new_constraint = [var for var, coeff in constraint_as_dict.items()
                               if coeff==-1][0]
            var_combinations = sorted(
                itermonomials(list(constraint.free_symbols), 2),
                key=monomial_key('lex', list(constraint.free_symbols))
            )
            new_constraint -= [term for term in var_combinations
                                if term.is_Mul][0]
            return new_constraint, True
        if kind == 'eq' :
            var_combinations = sorted(
                itermonomials(list(constraint.free_symbols), 2),
                key=monomial_key('lex', list(constraint.free_symbols))
            )
            new_constraint = 0
            for term in var_combinations:
                if term.is_symbol:
                    new_constraint += term
                elif term.is_Mul:
                    new_constraint -= 2*term
        return new_constraint, True
    ### x + y >= 1, x + y = 1
    if all(condition_eq_5_6):
        var_combinations = sorted(
            itermonomials(list(constraint.free_symbols), 2),
            key=monomial_key('lex', list(constraint.free_symbols))
        )
        new_constraint = 0
        for term in var_combinations:
            if term.is_integer or term.is_Mul:
                new_constraint += term
            if term.is_Mul and kind=='eq':
                new_constraint += term
            if term.is_symbol and kind=='gt':
                new_constraint -= term
            if term.is_symbol and kind == 'eq':
                new_constraint -= 2*term
        return new_constraint, True

    return constraint, False

def penalty_symbols_as_set(penalties_array):
    """Return the penalties as a set
    """
    penalties = set()
    for p in penalties_array:
        if isinstance(p, Symbol):
            penalties = penalties.union(set({p}))
    return penalties

def subs_matrix(matrix, input):
    """Subs the symbols in the matrix with the given input for a matrix
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if isinstance(matrix[i, j], (Symbol, Add, Mul, Pow)):
                matrix[i, j] = matrix[i, j].subs(input)
    return matrix

def simplify(polynom):
    """Simplifies a function with binary variables
    """
    polynom = Poly(polynom)
    new_polynom = 0
    variables = list(polynom.free_symbols)

    for var_i in variables:
        coefficient_i = polynom.as_expr().coeff(var_i)/2
        coefficient_i += polynom.as_expr().coeff(var_i ** 2)
        new_polynom += coefficient_i.as_coefficients_dict()[1] * var_i
        for var_j in variables:
            if var_j != var_i:
                coefficient_j = coefficient_i.coeff(var_j)
                new_polynom += coefficient_j.as_coefficients_dict()[1] *\
                                    var_i * var_j
    return new_polynom + polynom.as_expr().as_coefficients_dict()[1]
