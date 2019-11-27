# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:16:43 2019

@author: bdai729
"""
"""
reference: https://docs.sympy.org/latest/tutorial/solvers.html#tutorial-dsolve
"""
from sympy import *
init_printing(use_unicode=True,wrap_line=False, no_global=True)
from IPython.display import display # to display math formula image, print function only prints text

x, y, z = symbols('x y z')
# <codecell> A note about equation
print("Recall from the gotchas section of this tutorial that symbolic equations in SymPy are not represented by = or ==, but by Eq")
display(Eq(x,y))
print("Equation")
expr = Eq(x**2,1)
display(expr)
print("solve:")
display(solveset(expr,x))

print("instead of using x == y, you can just use x - y. For example")
expr = Eq(x**2-1,0)
display(expr)
print("Solve:")
display(solveset(expr,x))

print("OR skip Eq function solveset(x**2-1,x)")
expr = x**2-1
display(solveset(expr,x))

# <codecell> Solving Equations Algebraically
"""
The main function for solving algebraic equations is solveset. The syntax for solveset is solveset(equation, variable=None, domain=S.Complexes) Where equations may be in the form of Eq instances or expressions that are assumed to be equal to zero.

Please note that there is another function called solve which can also be used to solve equations. The syntax is solve(equations, variables) However, it is recommended to use solveset instead.

When solving a single equation, the output of solveset is a FiniteSet or an Interval or ImageSet of the solutions.
"""

print("When solving a single equation, the output of solveset is a FiniteSet or an Interval or ImageSet of the solutions")
expr = solveset(x**2-x,x)
display(expr)
expr = solveset(x-x,x,domain=S.Reals)
display(expr)
expr = solveset(sin(x)-1,x,domain=S.Reals)
display(expr)

# <codecell> Solving Differential Equations
"""
To solve differential equations, use dsolve. First, create an undefined function by passing cls=Function to the symbols function.
"""
f, g = symbols('f g', cls = Function)
print("Derivatives of f(x) are unevaluated")
expr = f(x).diff(x)
display(expr)
print("To represent the differential equation f′′(x)−2f′(x)+f(x)=sin(x), we would thus use")
leftSide = f(x).diff(x,x) - 2*f(x).diff(x) + f(x)
rightSide = sin(x)
diffeq = Eq(leftSide, rightSide)
display(diffeq)

print("To solve the ODE, pass it and the function to solve for to dsolve")
rslt = dsolve(diffeq,f(x))
display(rslt)

print("dsolve returns an instance of Eq. This is because in general, solutions to differential equations cannot be solved explicitly for the function")
rslt = dsolve(f(x).diff(x)*(1-sin(f(x)))-1,f(x))
display(rslt)
print("The arbitrary constants in the solutions from dsolve are symbols of the form C1, C2, C3, and so on")


# <codecell> play around as example
expr = x**2 / 4 / y * (x*y)**2 / (1 + (x*y)**2)
display(expr)
rslt = expr.diff(x)
display(rslt)
rslt = solveset(rslt, x)
print("max eff")
display(rslt)
