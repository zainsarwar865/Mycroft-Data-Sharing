# grandApricotComparison3.py
# Author: Vishal Kaushal
# Run as 'python grandApricotComparison3.py' to compare performance of 
# different alternatives of pybind type conversions listed in "methods" including apricot
# Uses python's timeit module

from sklearn.datasets import make_blobs
import random
import numpy as np
import submodlib.helper as helper
from submodlib.functions.facilityLocation import FacilityLocationFunction
from apricot import FacilityLocationSelection
import timeit
import csv

methods = ["fl_dense_py_kernel_np_numba_list", "fl_dense_py_kernel_np_numba_numpyarray", "fl_dense_py_kernel_np_numba_array", "fl_dense_py_kernel_np_numba_array32", "fl_dense_py_kernel_np_numba_array64", "apricot_dense"]

def fl_dense_py_kernel_np_numba_list():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="list")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def fl_dense_py_kernel_np_numba_numpyarray():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="numpyarray")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def fl_dense_py_kernel_np_numba_array():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="array")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def fl_dense_py_kernel_np_numba_array32():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="array32")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def fl_dense_py_kernel_np_numba_array64():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="array64")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def apricot_dense():
    obj = FacilityLocationSelection(n_samples=budget, metric='euclidean', optimizer='lazy')
    obj.fit_transform(dataArray)

cluster_std_dev = 2
num_executions = 3
num_places = 6
num_features = 1024
optimizer = 'LazyGreedy'

params = [(50, 5, 10), (100, 10, 50), (200, 10, 100), (500, 10, 100), (1000, 10, 100), (5000, 10, 100), (6000, 10, 100), (7000, 10, 100), (8000, 10, 100), (9000, 10, 100), (10000, 10, 100)]
#params = [(50, 5, 10), (100, 10, 50)]

first = True

results_csv = [["Num_Samples", "Function", "Time"]]
for param in params:
    print("Parameters: ", param)
    #prepare data to be used in the analysis
    num_samples = param[0]
    num_clusters = param[1]
    num_neighbors = param[2]
    budget = int(num_samples/10)

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    dataArray = np.array(data)

    if first == True:
        print("Pre compiling np_numba and apricot functions")
        fl_dense_py_kernel_np_numba_list()
        apricot_dense()
        first = False
    
    #K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    
    for method in methods:
        print("Method: ", method)
        row=[num_samples, method]
        func = method + "()"
        setup = "from __main__ import " + method
        t = timeit.timeit(func, setup, number=num_executions)
        t = round(t/num_executions,num_places)
        row.append(t)
        results_csv.append(row)
with open("submodlib_apricot_grand3.csv", "w") as f:
    writer = csv.writer(f)
    for result in results_csv:
        writer.writerow(result)
