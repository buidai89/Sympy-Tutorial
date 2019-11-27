# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:58:11 2019

@author: bdai729
"""
"""
# <codecell> reference
reference: https://docs.sympy.org/latest/tutorial/calculus.html

"""
from sympy import *
from IPython.display import display # to display math formula image, print function only prints text
init_printing(use_unicode = True,wrap_line=False, no_global=True)

# <codecell> DERIVATIVES
x,y,z = symbols('x y z')

diffCosX = diff(cos(x),x)
diffExpX = diff(exp(x**2),x)
display(diffCosX)
display(diffExpX)

diffThirdX = diff(x**4,x,x,x) # third derivative
diffThridX2 = diff(x**4,x,3) # third derivative
print("Derivatives")
display(diffThirdX)
display(diffThridX2)

expr = exp(x*y*z)
diffMultipleVariables =diff(expr,x,y,y,z,z,z,z) #derivative by x, then y 2 time and z 4 times sequentially
print("derivative by x, then y 2 time and z 4 times sequentially")
display(diffMultipleVariables)
# <codecell> call diff as a method
diffMultipleVariables = expr.diff(x,y,y,z,4)
display(diffMultipleVariables)

#To create an unevaluated derivative, use the Derivative class. It has the same syntax as diff.
deriv = Derivative(expr, x, y, x, 4)
print("create an unevaluated derivative, use the Derivative class.")
display(deriv) # type 'deriv' in the console, enter to see the unevaluated formula
derivEval = deriv.doit()
print("#To evaluate an unevaluated derivative, use the doit method.")
display(derivEval)
# <codecell> Derivative using tuple
m, n, a, b = symbols('m n a b')
expr = (a*x+b)**m
diffTuple = expr.diff((x,n))
print("Derivatives of unspecified order can be created using tuple (x, n) where n is the order of the derivative with respect to x")
display(diffTuple) # Type 'diffTuple' in the consol and enter to see the formula

# <codecell> INTEGRAL
x, y, z = symbols('x y z')
print("======================================================")
print("intergrals")
integralCosX = integrate(cos(x),x)
display(integralCosX)

print("To compute a definite integral, pass the argument")
integralDefinite = integrate(exp(-x),(x,0,oo))
display(integralDefinite)

print("As with indefinite integrals,(integration_variable, lower_limit, upper_limit)")
integralDefinite = integrate(exp(-x**2-y**2),(x,-oo,oo),(y,-oo,oo))
display(integralDefinite)

print("If integrate is unable to compute an integral, it returns an unevaluated Integral object")
expr = integrate(x**x,x)
display(expr)

# <codecell> call intergrate as a method
print("As with Derivative, you can create an unevaluated integral using Integral")
expr = Integral((log(x))**2,x)
display(expr)
print("to later evaluate this integral, call doit.")
display(expr.doit())

# <codecell> LIMITS
x, y, z = symbols('x y z')
print("SymPy can compute symbolic limits with the limit function")
expr = limit(sin(x)/x,x,0)
display(expr)

# <codecell> call limit as a method
print("Like Derivative and Integral, limit has an unevaluated counterpart, Limit. To evaluate it, use doit.")
expr = Limit(sin(x)/x,x,0)
display(expr)
display(expr.doit())

print("To evaluate a limit at one side only, pass '+' or '-' as a fourth argument to limit")
expr = Limit(1/x,x,0,'-')
display(expr)
display(expr.doit())

# <codecell> Series Expansison
print("SymPy can compute asymptotic series expansions of functions around a point")
expr = exp(sin(x))
display(expr)
display(expr.series(x,0,10))
print("The big 'O' means ommitted")
print("If you do not want the order term, use the removeO method.")
display(expr.series(x,0,4).removeO())
print("The O notation supports arbitrary limit points (other than 0):")
expr = exp(x-6).series(x,x0 = 6)
display(expr)

# <codecell> FINITE DIFFERENCES
print("The simplest way the differentiate using finite differences is to use the differentiate_finite function:")
f, g = symbols('f g', cls = Function)
expr = differentiate_finite(f(x)*g(x))
display(expr)

print("If we want to expand the intermediate derivative we may pass the flag evaluate=True:")
expr = differentiate_finite(f(x)*g(x), evaluate = True)
display(expr)
print("This form however does not respect the product rule.")

print("If you already have a Derivative instance, you can use the as_finite_difference method to generate approximations of the derivative to arbitrary order:")
f = Function('f')
dfdx = f(x).diff(x)
h = Symbol('h')
expr = dfdx.as_finite_difference()
display(expr)
# <codecell> MATRICES
expr = Matrix([1,2,3])
print("Matrix([1,2,3]) is a column vector")
display(expr)

expr = Matrix([[1,2,3]])
print("Matrix([[1,2,3]]) is a single row matrix")
display(expr)

print("m x n Matrix" )
expr = Matrix([[1, 2], [3, 4], [5, 6]])
display(expr)

# <codecell> Matrix Multiplication 
print("Matrices are manipulated just like any other object in SymPy or Python")
M = Matrix([[1, 2, 3], [3, 2, 1]])
N = Matrix([0, 1, 1])
expr = M*N
display(expr)

# <codecell> Matrix Shape
print("To get the shape of a matrix use shape")
M = Matrix([[1, 2, 3], [3, 2, 1]])
display(M.shape)

# <codecell> accessing rows and columns
print("accessing rows and columns")
M = Matrix([[1, 2, 3], [3, 2, 1]])
display(M.row(0))
display(M.row(1))
display(M.col(0))
display(M.col(2))

# <codecell> Delete/Inserting Rows and Cols
print("Delete/Inserting Rows and Cols")
M = Matrix([[1, 2, 3], [3, 2, 1]])
N = Matrix([[1, 2, 3]])
P = Matrix([1, 2, 3])
print("P = Matrix([1, 2, 3]) is consider as a vector, use N = Matrix([[1, 2, 3]]) instead")
display(P)
M.row_del(0)
display(M)
display(N)
M.row_insert(0, N)

print("Insert and delete methods do not operate in place, therefore the commands below can be used")
M = Matrix([[1, 2, 3]])
M = M.row_insert(1, Matrix([[4, 5, 6]]))
M = M.col_insert(0, Matrix([-1, -2]))
display(M)

# <codecell> Basic Methods + - * /
M = Matrix([[1, 3], [-2, 3]])
N = Matrix([[0, 3], [0, 7]])
display(M)
display(N)
print("M + N")
display(M + N)
print("M * N")
display(M * N)
print("3 * M")
display(3 * M)
print("M ** 2")
display(M ** 2)
print("inverse of M")
display(M**-1)
print("Transpose of M")
display(M.T)

# <codecell> identity matrices
print("identity matrices")
display(eye(3))
display(eye(4))

# <codecell> zero matrices
print("zero matrices")
display(zeros(3))
display(zeros(4))

# <codecell> Matrices of ones
print("Matrices of ones")
display(ones(3))
display(ones(2, 4))

# <codecell> diagonal matrices
print("The arguments to diag can be either numbers or matrices. A number is interpreted as a 1×1 matrix.")
expr = diag(1,2,3)
display(expr)

print("The matrices are stacked diagonally. The remaining elements are filled with 0s.")
expr = diag(-1, ones(2, 2), Matrix([5, 7, 5]))
display(expr)

# <codecell> Det - Determinant
print("To compute the determinant of a matrix, use det")
N = Matrix([[5, 2], [2, -1]])
display(N.det())

M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
display(M.det())

# <codecell> RREF
print("To put a matrix into reduced row echelon form, use rref. rref returns a tuple of two elements. The first is the reduced row echelon form, and the second is a tuple of indices of the pivot columns.")
M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
display(M.rref())

# <codecell> Nullspace
print("To find the nullspace of a matrix, use nullspace. nullspace returns a list of column vectors that span the nullspace of the matrix.")
M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
display(M)
display(M.nullspace())

# <codecell> Columnspace
print("To find the columnspace of a matrix, use columnspace. columnspace returns a list of column vectors that span the columnspace of the matrix.")
M = Matrix([[1, 2, 3, 0, 0], [4, 10, 0, 0, 1]])
display(M)
display(M.columnspace())

# <codecell> Eigenvalues, Eigenvectors, and Diagonalization
print("To find the eigenvalues of a matrix, use eigenvals. eigenvals returns a dictionary of eigenvalue:algebraic multiplicity pairs (similar to the output of roots).")
M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
display(M)
display(M.eigenvals())
display(M.eigenvects())

print("To diagonalize a matrix, use diagonalize. diagonalize returns a tuple (P,D), where D is diagonal and M=PDP**−1")
P, D = M.diagonalize()
display(P)
display(D)
print("double check the result")
display(P*D*P**(-1) == M)

print("If all you want is the characteristic polynomial, use charpoly. This is more efficient than eigenvals, because sometimes symbolic roots can be expensive to calculate.")
lamda = symbols('lamda')
p = M.charpoly(lamda)
expr = factor(p)
display(expr)

# <codecell> subscripts
Zin, Z1, Z2 = symbols('Z_in, Z_1, Z_2')
display(Zin, Z1, Z2)

xi1 = symbols('x_i^{(1)}')
display(xi1)

# <codecell> Latex, mathML, word
# if the system has not got latex, type pip install latex and restart kernal
from sympy import *
from IPython.display import display 
# to display math formula image, print function only prints text
from sympy.printing.mathml import print_mathml
init_printing(use_unicode = True,wrap_line=False, no_global=True)

Zin, Z1, Z2 = symbols('Z_in, Z_1, Z_2')
print(latex(Zin, Z1, Z2))
print_mathml(Zin)

xi1 = symbols('x_i^{(1)}')
print(latex(xi1))
print(xi1)

# in word, insert equation, choose latex on the design ribbon and paste the
# latex code in, magic will happen
# similarly the 'print' function outputs Unicode text, instead of selecting 
# 'latex' in the word's equation design ribbon, selectin 'Unicode' and paste
# the string in and see the magic.

