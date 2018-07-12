# This file is part of libigl, a simple c++ geometry processing library.
#
# Copyright (C) 2017 Sebastian Koch <s.koch@tu-berlin.de> and Daniele Panozzo <daniele.panozzo@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
import sys, os
import math
import time
import numpy as np
from iglhelpers import *
from numba import jit, autojit
import pyigl_proto as p


# Add the igl library to the modules search path
sys.path.insert(0, os.getcwd() + "/../")
import pyigl as igl

from shared import TUTORIAL_SHARED_PATH, check_dependencies

dependencies = []
check_dependencies(dependencies)

V = igl.eigen.MatrixXd()
U = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
L = igl.eigen.SparseMatrixd()

verbose = True

def timing(func, amount, text):
    times = []
    for i in range(amount):
        t0 = time.time()
        func()
        t1 = time.time()
        secs = t1 - t0
        times.append(secs)

    print(text, np.mean(times), np.std(times), np.median(times), "sec") if verbose else print(secs)

if True:
    #timing(lambda: igl.readOBJ(TUTORIAL_SHARED_PATH + "cube.obj", V, F), 100000, "Loading 100000 small objs:")

    #timing(lambda: igl.readOBJ(TUTORIAL_SHARED_PATH + "armadillo.obj", V, F), 20, "Loading 20 large objs:")

    #timing(lambda: igl.cotmatrix(V, F, L), 100, "Calculating 100 cotmatrices:")

#    def matrices():
#        matrix = np.random.randn(10000, 10000)
#        matrix = p2e(matrix)
#        matrix_2 = matrix * matrix

#    timing(lambda: matrices(), 5, "Calculating 50 matrix products:")

#    def matrices2():
#        matrix = igl.eigen.MatrixXd(10000, 10000)
#        matrix.setRandom()
#        matrix_2 = matrix * matrix

#    timing(lambda: matrices2(), 5, "Calculating 50 matrix products:")

#    def matrices(matrix):
#        matrix = p2e(matrix)
#        matrix_2 = matrix + matrix
#        return matrix_2
#    matrix = np.eye(10000)
#    matrix = matrix.astype(dtype="float64", order="F")
#    timing(lambda: matrices(matrix), 50, "Calculating 50 matrix products:")

#    def matrices1(matrix):
#        #matrix = p2e(matrix)
#        matrix_2 = matrix + matrix
#        return matrix_2
#    matrix = np.eye(10000)
#    matrix = matrix.astype(dtype="float64", order="F")
#    timing(lambda: matrices1(matrix), 50, "Calculating 50 matrix products:")

    def matrices2(matrix):
        #matrix = p2e(matrix)
        matrix_2 = p.matrix_add(matrix, matrix)
        return matrix_2
    matrix = np.eye(1000, dtype="float64")
#    matrix = matrix.astype(dtype="float32")#, order="C")
    timing(lambda: matrices2(matrix), 50, "Calculating 50 matrix products:")

#    def matrices2(matrix):
#        matrix_2 = matrix + matrix
#        return matrix_2

#    matrix = igl.eigen.MatrixXd(10000, 10000)
#    matrix.setIdentity()
#    timing(lambda: matrices2(matrix), 50, "Calculating 50 matrix products:")



#@autojit
#def matrices3():
#    matrix = np.random.randn(1000, 1000)
#    matrix = p2e(matrix)
#    su = 0
#    for i in range(1000):
#        for j in range(1000):
#            su += matrix[i, j]

#timing(lambda: matrices3(), 10, "Combined (np/eigen) matrix access:")


#@autojit
#def matrices4():
#    matrix = igl.eigen.MatrixXd(1000, 1000)
#    matrix.setRandom()
#    su = 0
#    for i in range(1000):
#        for j in range(1000):
#            su += matrix[i, j]

#timing(lambda: matrices4(), 10, "Eigen matrix access:")


#@autojit
#def matrices5():
#    matrix = np.random.randn(1000, 1000)
#    su = 0
#    for i in range(1000):
#        for j in range(1000):
#            su += matrix[i, j]

#timing(lambda: matrices5(), 10, "Numpy matrix access:")

##@autojit
#def matrices6():
#    matrix = np.random.randn(1000, 1000)
#    su = np.sum(matrix)

#timing(lambda: matrices6(), 10, "Numpy matrix access:")

#matrix = np.random.rand(1000, 1000)
#matrix_2 = matrix.dot(matrix)

#matrix_e = p2e(matrix)
#matrix_e2 = matrix_e * matrix_e

#for i in range(1000):
#    for j in range(1000):
#        a = matrix[i, j]

#for i in range(1000):
#    for j in range(1000):
#        a = matrix_e[i, j]



## Alternative construction of same Laplacian
#G = igl.eigen.SparseMatrixd()
#K = igl.eigen.SparseMatrixd()

## Gradient/Divergence
#igl.grad(V, F, G)

## Diagonal per-triangle "mass matrix"
#dblA = igl.eigen.MatrixXd()
#igl.doublearea(V, F, dblA)

## Place areas along diagonal #dim times

#T = (dblA.replicate(3, 1) * 0.5).asDiagonal() * 1

## Laplacian K built as discrete divergence of gradient or equivalently
## discrete Dirichelet energy Hessian

#temp = -G.transpose()
#K = -G.transpose() * T * G


## Use original normals as pseudo-colors
#N = igl.eigen.MatrixXd()
#igl.per_vertex_normals(V, F, N)
#C = N.rowwiseNormalized() * 0.5 + 0.5
