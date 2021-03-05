## Installation Notes

After downloading and unzipping the repository, you will be able to install the library using either native python setuptools: 
```
python setup.py install
```
or the pip package manager:
```
pip install .
```
This will also automatically download all required 3rd party packages.



## A Minimal Example
The following code snippet demonstrates how to use Quantum Circle's QUBO generator module `qc_qubogen`. 

More detailed Jupyter notebook examples can be found in the `/examples` directory, including descriptions of the functionality and an introduction to QUBO models.

```
>>> from sympy import Symbol
>>> from qc_qubogen import QUBOModel
>>> 
>>> 
>>> # 1. Create a symbolic binary variable wth sympy. Symbol
... x_0 = Symbol("x_0")
>>> 
>>> # 2. Define the function to be minimized
... objective = 2*x_0 + 3
>>> 
>>> # 3. Initialize the `QUBOModel` using the created objective function
... model = QUBOModel(objective)
>>> 
>>> # 4. Convert the problem and return the QUBO matrix and offset constant
... qubo_matrix, qubo_offset = model.model_to_qubo(return_offset=True)
>>> 
>>> print(f"QUBO matrix: Q = {qubo_matrix}")
QUBO matrix: Q = [[2.]]
>>> print(f"QUBO offset: c = {qubo_offset}")
QUBO offset: c = 3.0
```