import numpy as np
import math
from numpy import random
from scipy.fftpack import fft,ifft
def PowerMethod(Z,R,J):
    Qt,R=np.linalg.qr(np.dot(Z,R))
    for i in range(J):
        Qt,R=np.linalg.qr(np.dot(Z,np.dot(np.transpose(Z),Qt)))
    return Qt

def approxSVT(Z,R,lam,J):
    Q=PowerMethod(Z,R,J)
    [U,S,V]=np.linalg.svd(np.dot(np.transpose(Q),Z))
    (m,n)=np.shape(U)
    UU=[]
    VV=[]
    SS=[]
    if m>n:
        nn=n
    else:
        nn=m
    for i in range(nn):
        print(S[i],lam)
        if S[i]>lam:
            UU.append(U[:,i])
            VV.append(V[:,i])
            SS.append(S[i]-lam)
    print(np.shape(UU))
    # X=np.dot(Q,UU)
    # XX=np.dot(np.dot(X,S),np.transpose(V))
    # return UU,SS,VV,XX
    return UU,SS,VV
            
def rmse(target,prediction):
    error = []
    (m,n,k)=np.shape(target)
    for i in range(m):
        for j in range(n):
            for jj in range(k):
                error.append(target[i][j][jj] - prediction[i][j][jj])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    #print len(squaredError)
    rm=np.sqrt(sum(squaredError) / (m*n*k))
   # print sum(squaredError)
    return rm

# 直接生成低秩张量
def t_prod(A,B):
    [a1,a2,a3] = A.shape
    [b1,b2,b3] = B.shape
    A = fft(A)
    B = fft(B)
    C = np.zeros((a1,b2,b3),dtype=complex)
    for i in range(b3):
        C[:,:,i] = np.dot(A[:,:,i],B[:,:,i])
    C = ifft(C)
    return C

# n1代表节点，n2代表属性
# 返回图的张量
def create_graph(n1, n2, n3):
    graph = np.zeros((n1,n2,n3),dtype=complex)
    for i in range(n3):
        graph[:,:,i] = np.dot(random.rand(n1, 1),random.rand(1, n2))
    return graph

def create_adjacent(nNode,k,densityRate,unique):
    adjacent = np.zeros((nNode, nNode, k), dtype=complex)
    if unique:
        for i in range(k):
            # 生成唯一的邻接矩阵
            if i == 0:
                for iNode in range(math.floor((nNode + 1) / 2)):
                    for jNode in range(iNode + 1, nNode):
                        if random.rand(1) < densityRate:
                            adjacent[iNode, jNode, i] = 1
                            adjacent[jNode, iNode, i] = 1
            else:
                adjacent[:,:,i] = adjacent[:,:,0]
    else:
        for i in range(k):
            for iNode in range(math.floor((nNode + 1) / 2)):
                for jNode in range(iNode + 1, nNode):
                    if random.rand(1) < densityRate:
                        adjacent[iNode, jNode, i] = 1
                        adjacent[jNode, iNode, i] = 1
    return adjacent



# 根据邻接矩阵求UG的转置
def graph_fourier_transform_matrix(W):
    (nNode,nNode) = np.shape(W)
    D = np.zeros((nNode,nNode),dtype=complex)
    I = np.eye(nNode, dtype=complex)
    for iNode in range(nNode):
        D[iNode,iNode] = sum(W[iNode,:])

    D1 = np.zeros((nNode,nNode),dtype=complex)
    for iNode in range(nNode):
        if D[iNode,iNode] != 0:
            D1[iNode,iNode] = pow(D[iNode,iNode],-1/2)

    L = I - np.dot(np.dot(D1,W),D1)
    eigVal,eigVec = np.linalg.eig(L)

    # print('1111111',np.dot(np.dot(eigVec,np.diag(eigVal)),np.transpose(eigVec)))
    # print('22222', np.dot(np.dot(np.transpose(eigVec), np.diag(eigVal)), eigVec))
    # print('L',L)
    # exit(0)

    U = eigVec
    return U
