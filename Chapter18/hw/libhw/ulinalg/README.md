# Module umatrix, ulinalg

These are intended to be relatively small modules for use with MicroPython (Python3) which provide a minimal matrix class for representation, manipulation and a few linear algebra routines.

The matrix module is designed to offer close functional compatibility with 2-D Numpy arrays.

These routines are not designed to be particularly fast.

(eg. 3x3 matrix inversion takes about 350ms on a PyBoard.)

#### Files:

* __umatrix.py__ - matrix class.
* __ulinalg.py__ - supporting linear algebra routines (requires ```umatrix``` ).
* __ulinalg\_tests.py__ - testing file to check most of the features.

Currently supported: (see following sections and Properties, Methods and Functions for descriptions)

#### Provided by ```umatrix```

* assignment
* slicing (the third step argument is not supported)
* matrix/scaler element wise arithmetic operations
 * Note: __scaler OP matrix__ operations fail under MicroPython as reflected operations are not yet fully supported.
Operations need to be arranged in a __matrix OP scaler__ form. See the __Implementation Notes__ section.
* transpose
* iteration support
* sub-matrix assignment
* reshaping

#### Provided by ```ulinalg```

* eye
* ones
* determinant and inverse
* pseudo inverse (may need work)
* dot product
* cross product

<hr>

## Classes

```
umatrix.matrix
```
<hr>

#### Matrix instantiation

Supported matrix element types default to __bool__ and __int__. Support for __float__ and __complex__ are determined by the module upon importing
 and will depend on MicroPython compilation options and platform. 
All elements will be converted to the highest indexed type in the order of the above list (ie. if there is a __float__ in the data, all elements will be converted to __float__).
The result is held in ```umatrix.dtype```. The kwarg ```dtype=``` may also be used to force the type.

A matrix can be constructed using a list of lists representation, where each list member is a row in the matrix.
```
import umatrix
X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
```
Or by supplying a list of elements and the distance (strides) to move to get to the next column (```cstride```) and row (```rstride```). 
```
X = umatrix.matrix([0,1,2,3,4,5,6,7,8,9,10,11], cstride=1, rstride=3)
```

Both the above will result in:

```
X = mat([[0 , 1 , 2 ],
         [3 , 4 , 5 ],
         [6 , 7 , 8 ],
         [9 , 10, 11]])
```
Using:
```
X = umatrix.matrix([0,1,2,3,4,5,6,7,8,9,10,11], cstride=1, rstride=3, dtype=float)
```

results in:
```
X = mat([[0.0 , 1.0 , 2.0 ],
         [3.0 , 4.0 , 5.0 ],
         [6.0 , 7.0 , 8.0 ],
         [9.0 , 10.0, 11.0]])
```
<hr>

#### Matrix slicing

```
>>> import umatrix
>>> X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>> X
mat([[0 ,1 ,2 ],
     [3 ,4 ,5 ],
     [6 ,7 ,8 ],
     [9 ,10,11]])
>>>
```
```
>>> X[1,3]        # Returns a single element as int, float, complex
>>> 6
>>> 
```
```
>>> X[1]          # Returns a row matrix (Numpy returns a vector)
mat([[3, 4, 5]])
>>>
```
``` 
>>> X[1,:]        # Returns a row matrix (Numpy returns a vector)
mat([[3, 4, 5]])
>>>
```
```
>>> X[:,1]        # Returns a column matrix (Numpy returns a vector)
mat([[1 ],
     [4 ],
     [7 ],
     [10]])
>>>
```
```
>>> X[:,1:3]      # Returns a submatrix of every row by columns 1 and 2
mat([[1 , 2 ],
     [4 , 5 ],
     [7 , 8 ],
     [10, 11]])
>>>
```
```
>>> X[1:3,2:4]    # Returns a submatrix
mat([[5, 6],
     [8, 9]])
>>>

```
<hr>

#### Matrix assignment

Matrix elements can be assigned to using __bool__, __int__, __float__, __complex__, __list__ or from a __matrix__.
A list will assign elements in order from the source and wrap around if required.

```
>>>X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[1,:] = [20, 21, 22]
X
mat([[0 , 1 , 2 ],
     [20, 21, 22],
     [6 , 7 , 8 ],
     [9 , 10, 11]])
```
```
>>>X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[:,1] = [20, 21, 22]
X
mat([[0 , 20, 2 ],
     [3 , 21, 5 ],
     [6 , 22, 8 ],
     [9 , 20, 11]])  # wraps around to the first element of the source list
```
```
>>>X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[1:3,1:3] = [20, 21, 22, 23]
X
mat([[0 , 1 , 2 ],
     [3 , 20, 21],
     [6 , 22, 23],
     [9 , 10, 11]])
```
<hr>

#### Iteration

Iterating over a matrix will return the rows as a list of 1xn matrices (Numpy returns a list of vectors).

```
>>>X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>[i for i in X]
[mat([[0, 1, 2]]), mat([[3, 4, 5]]), mat([[6, 7, 8]]), mat([[9 , 10, 11]])]
```
Iterating over a slice of a matrix will return a list of elements.
```
>>>X = umatrix.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
[i for i in X[1,:]]
[3, 4, 5]
```
<hr>

## Implementation Notes

#### Differences from Numpy Arrays

* Slices are always a view in Numpy, in umatrix they are currently not a view
* Scaler as left hand side argument for [+,-,*,\,\\] operations are not supported in umatrix (see below) but a matrix as left hand side is.
* Single row/col slices are 1-D arrays in Numpy and  a 1xn (row) or nx1 (column) matrix in umatrix
* Numpy has a 0-d array, ```numpy.array(2)``` a special vector that acts like a scaler.
* umatrix doesn't support NaN, Inf, -Inf

#### Types

The ```umatrix``` module attempts to determine the supported types and floating point epsilon if __float__ is supported.

The results are held in ```umatrix.stypes``` and ```umatrix.flt_eps``` respectively.

The variable ```umatrix.dtype``` holds the default type used by some ```ulinalg``` routines (__float__ if supported) .

For example __flt\_eps__, __stypes__ under a few different platforms:
```
#    PyBoard         = 1.19E-7 , [<class 'bool'>, <class 'int'>, <class 'float'>, <class 'complex'>]
#    WiPy            = 1       , [<class 'bool'>, <class 'int'>]
#    Linux (uPy)     = 1.19E-7 , [<class 'bool'>, <class 'int'>, <class 'float'>, <class 'complex'>]
#    Linux (CPython) = 2.22E-16, [<class 'bool'>, <class 'int'>, <class 'float'>, <class 'complex'>]
```
Notes:
* uPy under Linux forces ```flt_eps = 1.19E-7``` since it appears that the underlying math functions use doubles but uPy operates with single precision.
* ```flt_eps``` is kind of irrelevant when all the work is done on one platform but the ```ulinalg_test``` file uses it to determine matrix equality.
<hr>

#### Matrix ```+,-,*,\,\\``` Scaler operations
Matrices used as the LH argument of a scaler operation will work for element wise operation.  Using a scaler as the RH argument does not work as reflected operations are not yet supported by MicroPython.

Negation of a matrix does work.

For example:

```
2+X     # does not work
X+2     # matrix plus element-wise addition works

2-X     # does not work
-X+2    # works

2*X     # does not work
X*2     # works
```
Scaler / matrix can be accomplished using the ```reciprocal(n=1)``` method.

```
2.0/X              # does not work
X.reciprocal()*2.0 # does work (1/x * 2.0)
```

The reason seems to be that the __MicroPython__ __int__ class __\_\_add\_\___ method (for example) does not raise __NotImplementedError__ and therefore the ```umatrix``` __\_\_radd\_\___ method is not invoked.
<hr>

#### Matrix equality
In MicroPython ```X == Y``` returns ```True``` if all elements of X and Y are equal and the they have the same shape.

The following functions are available:
* ```umatrix.matrix_equal(X, Y)```  - boolean indicating same data and shape.
* ```umatrix.matrix_equv(X, Y, tol=0)```  - boolean indication same data (within tol) and broadcastable.
* ```umatrix.matrix.isclose(X, Y, rtol=1.0E-5, atol=flt_eps)```  - boolean matrix indicating element wise equality (within tol).

Float and complex default to determining equality within ```flt_eps```.

In Numpy ```X == Y``` returns a boolean matrix indicating element equality.
To get a similar result to Numpy, use ```umatrix.matrix_isclose(X, Y)```.
<hr>

### Properties of umatrix.matrix
```
shape
```
> Returns the shape as a tuple (m, n).

```
shape = (p, q)
```
> In place shape change (not a copy)

```
is_square
```
> Returns __True__ if a square matrix

```
T
```
> Convenience property to return a transposed view

<hr>

### Methods of umatrix.matrix
```
copy()
```
> Returns a copy of the matrix

```
size(axis=0)
``` 
> Returns:

> axis=0 size (n*m) for Numpy compatibility

> axis=1 rows

> axis=2 columns

```
reshape(m, n)
``` 
> Returns a __copy__ of the matrix with a the shape (m, n)

```
transpose()
```
> Returns a __view__ of the matrix transpose

```
reciprocal(n=1)
```
> Returns a matrix with element wise reciprocals by default (or n/element). For use in scaler division. See __Implementation Notes__.

```
apply(func, *args, **kwargs)
```
> Call a scalar function on each element, returns a new matrix
passes *args and **kwargs to func unmodified
note: this is not useful for matrix-matrix operations

> e.g.

```
   y = x.apply(math.sin)
   y = x.apply(lambda a,b: a>b, 5) # equivalent to y = x > 5
   y = x.apply(operators.gt, 5)    # equivalent to y = x > 5 (not in MicroPython)
```

<hr>

### Functions provided by umatrix module
```
isclose(X, Y, rtol=1.0E-5, atol=flt_eps)
```
> Returns True if matrices X and Y have the same shape and the elements are within ```tol```.
> ```tol``` defaults to ```umatrix.flt_eps``` for use with ```int``` and to ```flt_eps``` for ```float``` and ```complex```.

> Ref. numpy.isclose()

```
matrix_equal(X, Y, tol=0)
```
> Returns a boolean matrix indicating element equality. X and Y must be the same shape.
> Similar to Numpys ```==```

```
matrix_equiv(X, Y)
```
> Returns a boolean indicating if X and Y share the same data and are broadcastable

<hr>

### Functions provided by ulinalg module
```
eps(a=1)
```
> Returns the closest floating point number within machine tolerance considered to be different from 'a'.

> Ref. numpy.spacing(), MATLAB/Octave eps() functions 

```
zeros(m, n, dtype=umatrix.ddtype)
```
> Returns a m x n matrix filled with zeros.

```
ones(m, n, dtype=umatrix.ddtype)
```
> Returns a m x n matrix filled with ones.

```
eye(m, dtype=umatrix.ddtype)
```
> Returns a m x m matrix with the diagonal of ones.

```
det_inv(X)
```
> Returns the determinant and inverse ```(d, i)``` of X if it exists.
> Uses Gaussian Elimination to get an upper triangular matrix.
>
> Returns (0, []) if the matrix is singular.
>
> ```Numpy.linalg``` provides separate functions for ```det``` and ```inv```.

```
pinv(X)
```
> Returns the results of the pseudo inverse operation of X.
>
> Not sure how robust this is but it works for at least one example.

```
dot(X, Y)
```
> The dot product operation.

```
cross(X, Y, axis=1)
```
> The cross product operation for 2x2 and 3x3 matrices.

> axis = 1 (default) is for Numpy compatibility.

> axis = 0 is for MATLAB, Octave, SciLab compatibility.
