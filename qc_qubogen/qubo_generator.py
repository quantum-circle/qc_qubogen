"""This module contains the QUBO Generator, which transforms constrained 
optimization problems into a QUBO Problem by given objective function 
and constraints. Also it computes the final QUBO matrix, 
which contains all the informations of the problem.
"""
import math
import warnings
from copy import deepcopy

import numpy as np
from sympy import Poly, Symbol

from .utils import *



def get_symbolic_binary_variables(n, start_idx=0):
    """Returns a list of 'n' (int) symbolic binary variables, 
    i.e. a list of sympy.Symbol 
    """
    return [Symbol(f"x{i+start_idx}") for i in range(n)]



class Constraint:
    def __init__(self, constraint, kind, penalty, slack_variables_start_idx=None,
                bounds = None, special=False):
        """Constaint object with some properties

        Parameters
        ----------
        constraint : sympy.core.expr.Expr or subclasses
            a constraint of the QUBO model
        kind : 'str'
            kind of constraint. 'lt' (less or equal than), 'gt' (greater or equal than)
            or equal constraint
        penalty : int or float
            panlty factor, which will be multiplied with the constraint
        slack_variables_start_idx : int, optional
            the amount of slack variables used before this constraint was added
            to the model, by default None
        bounds : tuple or list, optional
            should be of length 2 and gives the boundary of the constraint, by default None
        special : bool, optional
            the special constraints can be found in the paper "A Tutorial on Formulating
            and Using QUBO Models" by Fred Glover, Gary Kochenberger, Yu Du. By default False
        """
        self.constraint = constraint
        self.constraint_without_slack = constraint
        self.penalty = penalty
        self.special = special
        self.kind = kind
        self.bounds = bounds
        self.slack_variables_start_idx = slack_variables_start_idx
        

    @property
    def bounds(self):
        return self._bounds
    

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = None
        elif not isinstance(bounds, (tuple, list, np.ndarray)) and len(bounds)==2:
            raise ValueError('bounds must be a tuple, list or np.ndarray of length 2')
        else:
            self._bounds = bounds
            

    @property
    def variables(self):
        if self.constraint != 0:
            return self.constraint.free_symbols
        return set()
    

    @property
    def num_variables(self):
        return len(self.variables)
    

    @property
    def minimum(self):
        """The lower bound of the constraint, which is either the estimated minimum
        of the constraint or given by bounds.
        """
        minimum = sum(
            coeff for var, coeff in self.constraint.as_coefficients_dict().items()
            if coeff<0 and var != 1
        )
        if self.bounds is None or self.bounds[0] is None:
            return minimum
        if self.bounds[0] < minimum:
            warnings.warn(
                "The lower bound is smaller than the estimated minimum of "
                "the constraint. Therefore the lower bound is ignored."
            )
            return minimum
        return self.bounds[0]
    

    @property
    def slack(self):
        """The slack, which will be added to the constraint
        """
        maximum = -1 * float(self.constraint.as_coefficients_dict()[1])
        if self.kind == 'eq' or self.special:
            return None
        if self.bounds is not None and self.bounds[1] is not None:
            if self.bounds[1] > maximum:
                warnings.warn(
                    "The upper bound is higher than the right hand side of "
                    "the constraint (constant). Therefore the upper bound is ignored."
                )
            else:
                maximum = self.bounds[1]
        slack_var= maximum - self.minimum
        if slack_var < 0:
            warnings.warn(
                "The constraint can not be fulfilled. Therefore the constraint will "
                "not be added. Reason is either the right hand side of the constraint "
                "(constant) or the bounds, if added."
            )
            return 400
        if slack_var == 0:
            return None
        else:
            num_slack = int(math.log(math.ceil(slack_var), 2)) + 1

            coeff = np.array([2**i for i in range(num_slack)])
            slack_symbols = get_symbolic_binary_variables(num_slack, self.slack_variables_start_idx)
            if self.kind == 'gt':
                return - (slack_symbols @ coeff)
            return slack_symbols @ coeff
        

    @property
    def slack_variables(self):
        if self.slack is not None and self.slack != 400:
            return self.slack.free_symbols
        return None
    

    @property
    def num_slack_variables(self):
        return len(self.slack_variables)
    

    @property
    def offset(self):
        """The constant in the constraint
        """
        if self.constraint != 0:
            return float(self.constraint.as_expr().as_coefficients_dict()[1])
        return 0
    

    def __len__(self):
        return len(self.variables)
    

    def __repr__(self):
        return f"{self.__class__.__name__}(Size={len(self)})"



class QUBOModel:   
    def __init__(self, model=0):
        """

        Parameters
        ----------
        model : sympy.core.expr.Expr or subclasses
            should have as input the objective function
        """
        self.poly_model = model
        self.objective_function = deepcopy(model)
        self.constraints = []
        self.penalties = []
        self.qubo_matrix = None
        self.slacks = []
        

    @property
    def poly_model(self):
        """The function, which will be transformed into a qubo_matrix
        """
        return self._poly_model
    

    @poly_model.setter
    def poly_model(self, polynom):
        if isinstance(polynom, (int, float)):
            self._poly_model = polynom
        elif not Poly(polynom).is_quadratic:
            raise ValueError(
                "The poly_model is not quadratic and therefore "
                "cannot be transformed into a QUBO."
            )
        else:
            self._poly_model = Poly(polynom)
            

    @property
    def variables(self):
        """Set of variables in the QUBOModel Object
        """
        if not isinstance(self.poly_model, (int, float)):
            variables = self.poly_model.free_symbols
            penalties = penalty_symbols_as_set(self.penalties)
            if penalties:
                return variables.difference(penalties)
            return variables
        return set()
    

    @property
    def num_variables(self):
        return len(self.variables)
    

    @property
    def binary_variables(self):
        """Set of variables, without slack variables
        """
        binary_variables_temp = set()
        if self.objective_function != 0:
            binary_variables_temp = binary_variables_temp.union(
                                        self.objective_function.free_symbols)
        if not np.array(self.constraints).all():
            for cons in self.constraints:
                if cons is not None:
                    binary_variables_temp = binary_variables_temp.union(cons.variables)
            penalties = penalty_symbols_as_set(self.penalties)
            if penalties:
                return binary_variables_temp.difference(penalties)
            return binary_variables_temp
        return binary_variables_temp
    

    @property
    def num_binary_variables(self):
        return len(self.binary_variables)
    

    @property
    def slack_variables(self):
        if not np.array(self.slacks).all():
            return set()

        slack_variables_temp = set()
        for slack_cons in self.slacks:
            if slack_cons is not None:
                slack_variables_temp = slack_variables_temp.union(slack_cons.free_symbols)
        return slack_variables_temp
    

    @property
    def num_slack_variables(self):
        return len(self.slack_variables)
    

    @property
    def offset(self):
        """The constant in the poly_model, which will not be considered in the qubo_matrix
        """
        if self.poly_model != 0:
            return float(self._poly_model.as_expr().as_coefficients_dict()[1])
        return 0
    

    @property
    def offset_objective_function(self):
        if self.objective_function != 0:
            return float(self.objective_function.as_expr().as_coefficients_dict()[1])
        return 0
    

    def __len__(self):
        return len(self.variables)
    

    def __repr__(self):
        return f"{self.__class__.__name__}(Size={len(self)})"


    def model_to_qubo(self, return_offset=False):
        """Transforms the QUBO Model into a Matrix,
        which can be used as an input for various solver

        Returns
        -------
        matrix: np.array
            QUBO Matrix of the model.
        """
        if isinstance(self.poly_model, (float, int)):
            return 0
        penalties = penalty_symbols_as_set(self.penalties)
        variables = sort_variables(self.variables.difference(penalties))
        
        if penalties:
            matrix = np.zeros((self.num_variables, self.num_variables), dtype=object)
        else:
            matrix = np.zeros((self.num_variables, self.num_variables))

        for var_i_index, var_i in enumerate(variables):
            coefficient_i = self.poly_model.as_expr().coeff(var_i)
            for degree in range(self.poly_model.degree()-1):
                coefficient_i += self.poly_model.as_expr().coeff(var_i ** (degree+2))
            if coefficient_i.is_Float:
                matrix[(var_i_index, var_i_index)] = float(coefficient_i)
            elif coefficient_i.free_symbols.issubset(penalties):
                matrix[(var_i_index, var_i_index)] = coefficient_i
            else:
                matrix[(var_i_index, var_i_index)] = float(
                    coefficient_i.as_coefficients_dict()[1]
                )
            for var_j_index, var_j in enumerate(variables):
                if var_j != var_i:
                    coefficient_j = coefficient_i.coeff(var_j)
                    if coefficient_j.is_Float:
                        matrix[(var_i_index, var_j_index)] = float(coefficient_j)/2
                        matrix[(var_j_index, var_i_index)] = float(coefficient_j)/2
                    elif coefficient_j.free_symbols.issubset(penalties):
                        matrix[(var_i_index, var_j_index)] = coefficient_j/2
                        matrix[(var_j_index, var_i_index)] = coefficient_j/2
                    else:
                        matrix[(var_i_index, var_j_index)] = float(
                            coefficient_j.as_coefficients_dict()[1]
                        )/2
                        matrix[(var_j_index, var_i_index)] = float(
                            coefficient_j.as_coefficients_dict()[1]
                        )/2

        self.qubo_matrix = matrix
        if return_offset:
            return self.qubo_matrix, self.offset
        return matrix


    def add_constraint(self, constraint, kind, penalty=None, bounds=None):
        """Adds Constraint to the objective function or the model.
        If a less than or greater than constraint is added, the corresponding
        slack variable doesnt consider any other constraint.
        The slackvariable has the symbol name "_z".

        Parameters
        ----------
        constraint : sympy
            constraint, which should be added to the objective function
        kind : string
            'eq' (equal), 'gt' (greater or equal than) or 'lt' 
            (less or equal than) (in)equality
        penalty : float or symbol, optional
            Penalty factor for the constraint, 
            by default: upper - lower bound of the constraint
        bounds : tuple, optional
            lower and upper bound for the constraint, by default None

        Returns
        -------
        sympy
            poly_model + penalty * (constraint ** 2)

        Raises
        ------
        ValueError
            is raised if the constraint is not linear. 
            Otherwise it cannot transformed into a QUBO
        NameError
            is raised if neither of the kinds above is given for kind
        """
        if not Poly(constraint).is_linear:
            raise ValueError(
                "The Constraint is not linear and therefore it "
                "cannot be transformed into a QUBO"
            )

        if penalty is None:
            penalty = sum([abs(coef) for coef in Poly(constraint).coeffs()])

        constraint, special = special_constraint(constraint, kind)
        if special or kind =='eq':
            if bounds is not None:
                warnings.warn(
                    "The bounds will be ignored, since there are no "
                    "slack variables required."
                )
            if kind == 'eq':
                cons = Constraint(constraint, kind, penalty)
                self.poly_model = self.poly_model + cons.penalty * cons.constraint**2
            cons = Constraint(constraint, kind, penalty, special)
            self.poly_model = self.poly_model + cons.penalty * cons.constraint

            self.constraints = np.append(self.constraints, cons)
            self.slacks = np.append(self.slacks,cons.slack)
            self.penalties = np.append(self.penalties, cons.penalty)
            return cons

        elif kind in ('lt', 'gt'):
            if kind == 'gt':
                constraint = -1 * constraint
            if bounds is None:
                cons = Constraint(
                    constraint, 
                    kind, 
                    penalty,
                    slack_variables_start_idx=self.num_variables
                )
            else:
                cons = Constraint(
                    constraint, 
                    kind, 
                    penalty,
                    slack_variables_start_idx=self.num_variables, 
                    bounds=bounds
                )

            if cons.slack == 400:
                return self.poly_model
            if cons.slack == 0:
                warnings.warn('The constraint is an equation')
                self.poly_model = self.poly_model + cons.penalty * cons.constraint**2
                self.slacks = np.append(self.slacks, cons.slack)
                self.constraints = np.append(self.constraints, cons)
                self.penalties = np.append(self.penalties, cons.penalty)
                return cons

            if kind == 'gt':
                cons.constraint -= cons.slack
            else:
                cons.constraint += cons.slack

            self.constraints = np.append(self.constraints, cons)
            self.poly_model = self.poly_model + penalty * (cons.constraint ** 2)
            self.slacks = np.append(self.slacks, cons.slack)
            self.penalties = np.append(self.penalties, cons.penalty)
            return cons
        else:
            raise NameError('Unknown kind of inequality')


    def evaluate_objective_function(self, bitstring):
        """Evaluates the objective function for the bitstring

        Parameters
        ----------
        bitstring : np.array
            a bistring array with 0 and 1

        Returns
        -------
        float
            solution of objective function at the point bitstring

        Raises
        ------
        ValueError
            if the length of bitstring is not the same as the amount of variables
            without slack variables
            if bitstring has an element, which is non zero or not one
        """
        if any(bit != 0 or bit != 1 for bit in bitstring):
            raise ValueError("bitstring must only of 0 and 1s")
        if not (self.num_binary_variables == len(bitstring) or self.num_variables == len(bitstring)):
            raise ValueError("The shape of the solution and the model does not fit.")
        return bitstring @ self.qubo_matrix[0:len(bitstring), 0:len(bitstring)] @ bitstring \
                + self.offset_objective_function


    def evaluate_model(self, bitstring):
        """Evaluates the model, which includes the constraints.

        Parameters
        ----------
        bitstring : np.array
            a bistring array with 0 and 1

        Returns
        -------
        float
            solution of model at the point bitstring

        Raises
        ------
        ValueError
            if the length of bitstring is not the same as the amount of variables of the model
            if bitstring has an element, which is non zero or not one
        """
        if any(bit != 0 or bit != 1 for bit in bitstring):
            raise ValueError("bitstring must only of 0 and 1s")
        if self.num_variables != len(bitstring):
            raise ValueError("The shape of the solution and the model does not fit")
        return bitstring @ self.qubo_matrix @ bitstring + self.offset
    

    def subs(self, input, inplace=False):
        """Substitues the sympy symbols in the model with the input

        Parameters
        ----------
        input : array
            The array should contain tuples with one symbol and one int or float.
        inplace : bool, optional
            if true, the whole model will be subsituted, otherwise a copy will
            be created. By default False

        Returns
        -------
        QUBOModel
            The return is the QUBOModel object
        """

        if inplace:
            self.poly_model = self.poly_model.subs(input)
            for i, penalty in enumerate(self.penalties):
                if isinstance(penalty, Symbol):
                    self.penalties[i] = self.penalties[i].subs(input)
            if self.qubo_matrix is not None:
                self.qubo_matrix = subs_matrix(self.qubo_matrix, input)
                self.qubo_matrix.astype('float')
            return self

        QUBOModel_copy = deepcopy(self)
        QUBOModel_copy.poly_model = QUBOModel_copy.poly_model.subs(input)
        for i, penalty in enumerate(QUBOModel_copy.penalties):
            if isinstance(penalty, Symbol):
                QUBOModel_copy.penalties[i] = QUBOModel_copy.penalties[i].subs(input)
        if QUBOModel_copy.qubo_matrix is not None:
            QUBOModel_copy.qubo_matrix = subs_matrix(QUBOModel_copy.qubo_matrix, input)
            QUBOModel_copy.qubo_matrix.astype('float')
        return QUBOModel_copy



def from_matrix_to_qubo(matrix):
    """This method transforms a matrix to a QUBOModel Object

    Parameters
    ----------
    matrix : list, numpy array
        QUBO Matrix

    Returns
    -------
    QUBOModel
        Object QUBOModel from the given QUBO Matrix
    """
    if not np.array_equal(np.array(matrix), np.array(matrix).T):
        raise ValueError(
            "The matrix is not quadratic, therefore cannot be "
            "transformed into a QUBO Model"
        )
    symbolic_var = get_symbolic_binary_variables(len(matrix))
    poly_model = symbolic_var @ np.array(matrix) @ symbolic_var
    return QUBOModel(poly_model)