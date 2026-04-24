import math
from typing import Callable, Dict, List, Tuple

import numpy as np


SAFE_FUNCS = { # dictionary for math operations
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'exp': np.exp,
    'log': np.log,
    'ln': np.log,
    'log10': np.log10,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'pi': np.pi,
    'e': np.e,
    'power': np.power,
    'floor': np.floor,
    'ceil': np.ceil,
}


class NumericalMethodError(Exception): # custom error type
    pass


# ---------- Linear algebra ------------------------------------------------------------
def parse_matrix(matrix_text: str) -> np.ndarray: # convert text matrix to numpy array
    """
    Convert a user-entered matrix string into a NumPy array.
    Rows should be separated by semicolons, and values by spaces or commas.

    Example:
    1 2; 3 4
    or
    1,2; 3,4
    """
    try: # remove extra spaces ad split string matrix into rows
        rows = [row.strip() for row in matrix_text.strip().split(';') if row.strip()]
        matrix = []
        for row in rows:
            cleaned = row.replace(',', ' ') # replace commas with spaces
            values = [float(val) for val in cleaned.split()] # split row by spaces and convert each item to float
            matrix.append(values) # add row into full matrix
    except ValueError as exc: # error handling
        raise NumericalMethodError('Matrix entries must be valid numbers.') from exc

    if not matrix: 
        raise NumericalMethodError('Please enter a matrix.')

    row_lengths = {len(row) for row in matrix} # ensure every row has same # of columns
    if len(row_lengths) != 1:
        raise NumericalMethodError('Each matrix row must have the same number of entries.')

    arr = np.array(matrix, dtype=float) # turn python list into numpy array
    if arr.shape not in [(2, 2), (3, 3)]: # check matrix size & raise error if not 2x2 or 3x3
        raise NumericalMethodError('Only 2x2 and 3x3 matrices are supported.')

    return arr # return cleaned numpy array


def format_matrix(matrix: np.ndarray) -> str: # convert numpy matrix into formatted string for GUI display
    rows = []
    for row in matrix:
        rows.append('[' + '  '.join(f'{val:10.6f}' for val in row) + ']') # formatting
    return '\n'.join(rows) # join formatted rows inot one multiline string


def linear_algebra_operations(matrix_text: str) -> Dict[str, object]: # main algebra function
    A = parse_matrix(matrix_text) # convert text input into numpy array
    det = float(np.linalg.det(A)) # compute determinant of A
    results: Dict[str, object] = { # dictionary to store results
        'matrix': A,
        'shape': A.shape,
        'determinant': det,
        'transpose': A.T,
    }

    if abs(det) < 1e-12: # check if determinant is very close to 0
        results['inverse'] = None # these don't exist for singular matrix
        results['eigenvalues'] = None
        results['eigenvectors'] = None
        results['message'] = 'Matrix is singular, so the inverse does not exist.'
    else:
        results['inverse'] = np.linalg.inv(A) # compute inverse
        eigenvalues, eigenvectors = np.linalg.eig(A) # comute eigenvalues & eigenvectors
        results['eigenvalues'] = eigenvalues
        results['eigenvectors'] = eigenvectors
        results['message'] = 'Inverse and eigen information computed successfully.' 

    return results # return results dictionary


# ---------- Function parsing -------------------------------------------------------------------
def _clean_expr(expr: str) -> str: # remove extra spaces at the ends & replace ^ with **
    return expr.replace('^', '**').strip()


def build_scalar_function(expr: str) -> Callable[[float], float]: # return real python function of 1 var from string
    expr = _clean_expr(expr) # user-friendly exponents

    def f(x: float) -> float: # 
        allowed = dict(SAFE_FUNCS) # create environment eval() can see
        allowed['x'] = x 
        try:
            value = eval(expr, {'__builtins__': {}}, allowed) # evaluate user's text as python code
        except Exception as exc: # error handling
            raise NumericalMethodError(f'Could not evaluate f(x): {exc}') from exc
        return float(value) # return float

    return f # return inner function


def build_two_var_function(expr: str) -> Callable[[float, float], float]: # same function but for 2 variables
    expr = _clean_expr(expr)

    def f(x: float, y: float):
        allowed = dict(SAFE_FUNCS)
        allowed['x'] = x
        allowed['y'] = y
        try:
            value = eval(expr, {'__builtins__': {}}, allowed)
        except Exception as exc:
            raise NumericalMethodError(f'Could not evaluate f(x, y): {exc}') from exc
        return value

    return f


# ---------- Root finding --------------------------------------------------------------------------------------------------------------
def bisection_method(expr: str, a: float, b: float, tol: float, max_iter: int) -> Tuple[float, int, List[Dict[str, float]]]: # bisection solver
    f = build_scalar_function(expr) # convert user string into callable function
    fa = f(a) # compute function values
    fb = f(b)
    if fa * fb > 0: # ensure opposite signs
        raise NumericalMethodError('Bisection needs f(a) and f(b) to have opposite signs.')

    history = []
    c = a
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0 # midpoint
        fc = f(c) # evaluate at midpoint
        history.append({'iter': i, 'a': a, 'b': b, 'c': c, 'f(c)': fc}) # store iteration info

        if abs(fc) < tol or 0.5 * abs(b - a) < tol: # stop if func value close enough to 0 or interval is small enough
            return c, i, history

        if fa * fc < 0: # keep the half-interval that contains the sign change
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, max_iter, history # if method never meets tolerance, return best estimate


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-6) -> float: # approximate derivative
    return (f(x + h) - f(x - h)) / (2.0 * h)

# implement newton-raphson
def newton_method(expr: str, x0: float, tol: float, max_iter: int, h: float = 1e-6) -> Tuple[float, int, List[Dict[str, float]]]:
    f = build_scalar_function(expr) # create f(x) from user input
    x = x0 # start at intial guess
    history = [] 

    for i in range(1, max_iter + 1):
        fx = f(x) # evaluate function at current value
        dfx = central_difference(f, x, h) # approximate derivative 
        if abs(dfx) < 1e-14: # check for small derivative
            raise NumericalMethodError('Derivative became too small in Newton\'s method.')

        x_new = x - fx / dfx # newton-raphson formula
        history.append({'iter': i, 'x_n': x, 'f(x_n)': fx, "f'(x_n)": dfx, 'x_next': x_new}) # store iteration data for display

        if abs(x_new - x) < tol or abs(fx) < tol: # stop if new guess is close to old guess/function value is close to 0
            return x_new, i, history
        x = x_new # move to next step

    return x, max_iter, history # return best guess if tolerance not met

# implement secant method
def secant_method(expr: str, x0: float, x1: float, tol: float, max_iter: int) -> Tuple[float, int, List[Dict[str, float]]]:
    f = build_scalar_function(expr)
    history = []

    for i in range(1, max_iter + 1):
        fx0 = f(x0) # evaluate function at 2 guesses
        fx1 = f(x1)
        denom = fx1 - fx0 # denominator for secant formula
        if abs(denom) < 1e-14: # check for small denominator 
            raise NumericalMethodError('Secant denominator became too small.')

        x2 = x1 - fx1 * (x1 - x0) / denom # secant formula
        history.append({'iter': i, 'x_prev': x0, 'x_curr': x1, 'x_next': x2, 'f(x_curr)': fx1}) # store method progress

        if abs(x2 - x1) < tol or abs(f(x2)) < tol: # stop if consectuive guesses are close/function value @ new guess is close to 0
            return x2, i, history

        x0, x1 = x1, x2 # move current guess to previous & new guess to current

    return x1, max_iter, history # return last available guess


# ---------- Differentiation -------------------------------------------------------
def derivative_approx(expr: str, x0: float, h: float) -> Dict[str, float]: # compute 3 derivative approximations @ a point
    f = build_scalar_function(expr) # function from user input
    forward = (f(x0 + h) - f(x0)) / h # forward difference formula
    backward = (f(x0) - f(x0 - h)) / h # backward difference
    central = (f(x0 + h) - f(x0 - h)) / (2.0 * h) # central difference
    return { # return all computed values for GUI
        'forward': float(forward),
        'backward': float(backward),
        'central': float(central),
        'f(x0)': float(f(x0)),
    }


# ---------- Nonlinear systems --------------------------------------------------------
def numerical_jacobian(f1, f2, x: float, y: float, h: float = 1e-6) -> np.ndarray: # approximate jacobian matrix
    j11 = (f1(x + h, y) - f1(x - h, y)) / (2.0 * h) # approximate partial derivatives (f1/x)
    j12 = (f1(x, y + h) - f1(x, y - h)) / (2.0 * h) # f1/y
    j21 = (f2(x + h, y) - f2(x - h, y)) / (2.0 * h) # f2/x
    j22 = (f2(x, y + h) - f2(x, y - h)) / (2.0 * h) # f2/y
    return np.array([[j11, j12], [j21, j22]], dtype=float) # build & return 2x2 matrix

# solve nonlinear system of 2 equations w/ Newton's method
def newton_system_method(expr1: str, expr2: str, x0: float, y0: float, tol: float, max_iter: int):
    f1 = build_two_var_function(expr1) # turn both strings into callable functions
    f2 = build_two_var_function(expr2)
    x, y = float(x0), float(y0) # start iteration at given point
    history = []

    for i in range(1, max_iter + 1):
        F = np.array([float(f1(x, y)), float(f2(x, y))], dtype=float) # build function vector
        J = numerical_jacobian(f1, f2, x, y) # approximate j matrix at current point

        try: # solve linear system
            delta = np.linalg.solve(J, -F) # newton correction step
        except np.linalg.LinAlgError as exc: # handle singular matrix
            raise NumericalMethodError('Jacobian is singular or nearly singular.') from exc

        x_new = x + delta[0] # update new point
        y_new = y + delta[1]
        history.append({ # store history for method steps
            'iter': i,
            'x': x,
            'y': y,
            'f1': F[0],
            'f2': F[1],
            'dx': delta[0],
            'dy': delta[1],
        })

        # stop if correction step or function values are very small
        if max(abs(delta[0]), abs(delta[1])) < tol or max(abs(F[0]), abs(F[1])) < tol:
            return (float(x_new), float(y_new)), i, history

        x, y = x_new, y_new # update current guess

    return (float(x), float(y)), max_iter, history # return if max iterations reached
