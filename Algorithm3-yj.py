import math
import json
import numpy as np
from numpy import random
# from numpy import linalg as la
from scipy import linalg as la
from scipy.fftpack import fft,ifft
from federated import approxSVT,t_prod,create_graph,create_adjacent,graph_fourier_transform_matrix

X_List = []
V_List = []
F_List = []
c_list = []
Omega_list = []
data_predict = None
PredictWithParameter = [] # [{Para:[epsilon,delta,omega_percent,T,lamda],Error:Float}]

def SVT(X,lam):
    [U, S, V] = np.linalg.svd(X)
    (m) = np.shape(S)
    UU = []
    VV = []
    SS = []
    index = 0
    for i in range(int(m[0])):
        if S[i] > lam:
            UU.append(U[:, i])
            VV.append(V[:, i])
            SS.append(S[i] - lam)
            index += 1
    return UU, SS, VV,index
#Differentially Private Federated Tensor Completion
# np.zero((x,y,z))
def F_global(X,adjacent,k,sigma,L,T,J): # dp_para:[epsilon,delta]
    (m,n,k) = np.shape(X)
    ListInit(T,m,n,k)
    U = np.zeros((m,m,k),dtype=complex)
    V = np.zeros((n,n,k),dtype=complex)
    A = np.zeros((m,n,k),dtype=complex)
    lamda_temp = 5
    lamda = 1
    IterData['Para'].extend([lamda,lamda_temp])
    for t in range(T+1):
       # lamda_temp=np.lingla.svd()
        for i in range(1,k+1):
            u = U[:,0,i-1]
            v = V[:,0,i-1]
            A[:,:,i-1],ss= F_local(lamda_temp,lamda,sigma,i,u,v,t,T,L,J,Pomega(X[:,:,i-1]),adjacent[:,:,i-1])
            # if np.shape(ss)[0] == 0:
            #     lamda = 0
            # else:
            #     lamda=sum(ss)/(np.shape(ss)[0])
            # lamda_temp=lamda+1
        if t < T:
            U,S,V = t_svd(A)

def ListInit(T,m,n,k):
    global X_List,V_List,F_List,c_list,data_predict
    # X_List.clear()
    # V_List.clear()
    # F_List.clear()
    # c_list.clear()
    X_List=[]
    V_List=[]
    F_List=[]
    c_list=[]
    for i in range(k):
        X_List.append([np.zeros((m,n), dtype=complex)]*(T+2))
        V_List.append([np.zeros((m,n), dtype=complex)]*(T+2))
        F_List.append([np.zeros((m,n), dtype=complex)]*(T+2))
        c_list.append(2)
    data_predict = np.zeros((m, n, k), dtype=complex)
    for i in range(1,k+1):
        X0 = np.zeros((m, n), dtype=complex)
        X1 = np.zeros((m, n), dtype=complex)
        X_List[i-1][0] = X0
        X_List[i-1][1] = X1

        V0 = np.zeros((n, n), dtype=complex)
        V1 = np.zeros((n, n), dtype=complex)
        V_List[i-1][0] = V0
        V_List[i-1][1] = V1


def F_local(lamda_temp,lamda,sigma,i,u,v,t,T,L,J,X_observed,adjacent_i):
    global X_List,V_List,F_List,c_list,data_predict
    (m,n) = np.shape(X_observed)
    c = c_list[i-1]
    theta = (c-1)/(c+2)
    mu=2
    tau = 20
    # U,s,V = la.svd(Xt)
    #lamda = max(s)
   # lamda_temp = lamda

    temp_v = random.uniform(0,1)
    while temp_v == 0:
        temp_v = random.uniform(0, 1)


    lamda_t = (lamda_temp-lamda)*temp_v+lamda

    E = random.normal(loc=0, scale=sigma, size=(m,n))

    # UG = np.eye(m,dtype=complex)
    # UG_tran = np.transpose(UG)
    UG = graph_fourier_transform_matrix(adjacent_i)
    UG_tran = np.transpose(UG)
    # UG_tran = UG

    Xt = np.dot(UG_tran,X_List[i-1][t])
    if t == 0:
        Xt_sub1 = np.zeros((m, n), dtype=complex)
    else:
        Xt_sub1 = np.dot(UG_tran,X_List[i-1][t-1])

    Y = Xt+theta*(Xt-Xt_sub1)

    uu = []
    uu.append(u)
    vv = []
    vv.append(v)
    Z = Y-mu*tau*np.dot(np.transpose(uu),vv)
    # if i == 1:
    #     print('u*v:',np.dot(np.transpose(uu),vv))
    # Q,R = la.qr(np.dot(V_List[i-1][t],V_List[i-1][t-1]))

    # Ut_add1,St_add1,Vt_add1 = approxSVT(Z,R,mu*lamda_t,J=J)
    lam=mu*lamda_t
    Ut_add1, St_add1, Vt_add1,index = SVT(Z,lam)
    # print(Ut_add1,St_add1,Vt_add1)
    V_List[i-1][t+1] = Vt_add1

    St_sigma = np.zeros((index, index),dtype=complex)
    for temp in range(index):
        St_sigma[temp, temp] = St_add1[temp]

    Xt_add1 = np.dot(np.transpose(Ut_add1),np.dot(St_sigma,Vt_add1))
    Xt_add1 = np.dot(UG,Xt_add1)
    X_List[i-1][t+1] = Xt_add1


    Xt_S = sum(St_add1)
    F_Xt = (la.norm(Pomega(X_List[i-1][t]-X_observed),ord=2)/2)+lamda*Xt_S
    F_List[i-1][t] = F_Xt

    if t > 1:
        if F_List[i-1][t] > F_List[i-1][t-1]:
            c_list[i-1] = 1
        else:
            c_list[i-1] = c+1
    if t == T:
        data_predict[:,:,i-1] = Xt_add1
        return np.zeros((m, n), dtype=complex),St_add1
    else:
        return Y-np.dot(UG_tran,X_observed)+E,St_add1



def t_svd(M):
    [n1, n2, n3] = M.shape
    D = np.zeros((n1, n2, n3), dtype=complex)
    D = fft(M)
    Uf = np.zeros((n1, n1, n3), dtype=complex)
    Thetaf = np.zeros((n1, n2, n3), dtype=complex)
    Vf = np.zeros((n2, n2, n3), dtype=complex)
    for i in range(n3):
        temp_U, temp_Theta, temp_V = la.svd(D[:, :, i], full_matrices=True)
        Uf[:, :, i] = temp_U
        Thetaf[:n2, :n2, i] = np.diag(temp_Theta)
        Vf[:, :, i] = temp_V
    U = np.zeros((n1, n1, n3))
    Theta = np.zeros((n1, n2, n3))
    V = np.zeros((n2, n2, n3))
    U = ifft(Uf).real
    Theta = ifft(Thetaf).real
    V = ifft(Vf).real
    return U, Theta, V

def t_svd_me(M):
    [k, m, n] = M.shape
    D = np.zeros((k, m, n), dtype=complex)
    D = fft(M)
    Uf = np.zeros((k, m, m), dtype=complex)
    Thetaf = np.zeros((k, m, n), dtype=complex)
    Vf = np.zeros((k, n, n), dtype=complex)
    for i in range(k):
        temp_U, temp_Theta, temp_V = la.svd(D[i, :, :], full_matrices=True)
        Uf[i, :, :] = temp_U
        Thetaf[i, :n, :n] = np.diag(temp_Theta)
        Vf[i, :, :] = temp_V
    U = ifft(Uf).real
    Theta = ifft(Thetaf).real
    V = ifft(Vf).real
    return U, Theta, V



#
def Judge_epsilon_delta(e,d):
    if e <= 2*math.log((1/d),10):
        return 1
    else:
        return 0



def BoundPomege(X):
    L_bound = 0
    (m,n,k) = np.shape(X)
    for i in range(k):
        X_P = Pomega(X[:,:,i])
        L = la.norm(X_P,ord=2)
        L_bound = max(L,L_bound)
    return L_bound


def Pomega(matrix):
    global Omega_list
    (m,n) = np.shape(matrix)
    X_P = np.zeros((m,n),dtype=complex)
    for index in Omega_list:
        i,j = index
        X_P[i,j] = matrix[i,j]
    return X_P


def GetGlobalOmega(m,n,per):
    global Omega_list
    Omega_list = []
    num = int(m*n*per)
    for tt in range(num):
        i = random.randint(0,m)
        j = random.randint(0,n)
        while [i,j] in Omega_list:
            i = random.randint(0,m)
            j = random.randint(0,n)
        Omega_list.append([i,j])


def JsonFileSave(Data,filename):
    path = 'D:/DOCYJ/Federate_learn_predict/' + filename
    file = open(path, "w+", encoding="UTF-8")
    file.writelines(Data)
    file.close()

def JsonFileLoad(filename):
    DataLoad = []
    path = 'D:/DOCYJ/Federate_learn_predict/' + filename
    file = open(path, "r", encoding="UTF-8")
    lines = file.readlines()
    for line in lines:
        DataLoad.append(json.loads(line))
    file.close()
    return DataLoad

def DealFile(filename):
    path = 'D:/DOCYJ/' + filename
    file = open(path, "r", encoding="UTF-8")
    file_w = open('D:/DOCYJ/Federate_learn_predict/matrix_graph_adjacent_unique_0.json',"w",encoding="UTF-8")
    line = file.readline()
    Dataload = line.split('}{')
    for i in range(len(Dataload)):
        if i == 0:
            Dataload[i] = Dataload[i] + '}\n'
        elif i == len(Dataload)-1:
            Dataload[i] = '{' + Dataload[i] + '\n'
        else:
            Dataload[i] = '{' + Dataload[i] + '}\n'
    file_w.writelines(Dataload)
    file.close()
    file_w.close()






if __name__ == "__main__":
    DataSave = []
    IterData = {'Para':[],'Error':0.0}
    epsilon = [0.1,0.2,0.5,1,2,5,6]
    delta = pow(10,-6)
    omega_percent = [n for n in np.arange(0.1, 1.1, 0.1)]
    T_list = [n for n in range(10,100,10)]
    m = 60
    n = 60
    k = 20
    r = 1
    error = [0]*(k)
    # graph = create_graph(m, n, k)
    graph = t_prod(random.rand(m,r,k),random.rand(r,n,k))
    adjacent = create_adjacent(m,k, densityRate=0.01,unique=0)
    # adjacent = create_adjacent(m,k,densityRate=0.01,unique=1)
    for e in epsilon:
        for per in omega_percent:
            GetGlobalOmega(m, n, per)
            L = BoundPomege(graph)
            if Judge_epsilon_delta(e,delta):
                dp_para = [e, delta]
                for T in T_list:
                    sigma = (4 * L * math.sqrt(2 * T * math.log((1 / dp_para[1]), 10))) / dp_para[0]
                    IterData['Para'] = [e,delta,per,T]
                    F_global(graph,adjacent,k,sigma,L,T,J=10)
                    for i in range(k):
                        error[i] = la.norm(graph[:,:,i]-np.abs(data_predict[:,:,i]),ord=2)/(m*n*per)
                    IterData['Error'] = sum(error)/k
                    PredictWithParameter.append(IterData)
                    DataSave.append(json.dumps(IterData) + '\n')
                    print(IterData)
    # JsonFileSave(DataSave,'matrix_graph_adjacent_unique_0.json')
    # JsonFileSave(DataSave, 'matrix_graph_adjacent_unique_1.json')
    JsonFileSave(DataSave, 'tensor_graph_adjacent_unique_0.json')
    # JsonFileSave(DataSave, 'tensor_graph_adjacent_unique_1.json')
    print(PredictWithParameter)




