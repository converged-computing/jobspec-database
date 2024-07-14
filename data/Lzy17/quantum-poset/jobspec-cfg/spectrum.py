#from numba import jit, cuda
import random
import numpy as np
n = 5
print(2**(4*(n**2)))

#@jit
def tally(arr):
    out = [];
    for index1 in range(0, len(arr)):
        count = 1
        if(len(arr[index1]) == 0):
            continue
        else:
            for index2 in range(index1 + 1, len(arr)):
                if(len(arr[index1]) == len(arr[index2])):
                    if(len(arr[index1]) != 0):
                        same = True;
                        for item in range(0, len(arr[index1])):
                            if((arr[index1])[item] != (arr[index2])[item]):
                                same = False
                                break
                        if same:
                            arr[index2] = []
                            count = count + 1

            out.append(count)
    return out

#@jit
def main():
    plotdistinct = []
    plotest = []
    for i in range (12,30):
        Nsamples = 2**i
        polys = []
        for count in range (1, Nsamples):
            Clinks = np.zeros( (4*n, 4*n) )
            for i in range(0,n):
                jlist = []
                for k in range(0, n):
                    o = random.randrange(n + 1, 3*n, 1)
                    jlist.append(o)
                for j in range(1,len(jlist)):
                    Clinks[i][jlist[j]] = 1

            for i in range(3*n, 4*n):
                klist = []
                for k in range(0, n):
                    o = random.randrange(n + 1, 3*n, 1)
                    klist.append(o)
                for j in range(1,len(jlist)):
                    Clinks[klist[j]][i] = 1

            Csparse = Clinks
            for i in range(0,n):
                for j in range(3*n + 1, 4*n):
                    Npaths = 0
                    for k in range (n+1, 3*n):
                        Npaths = Npaths + Csparse[i][k]*Csparse[k][j]
                    if(Npaths > 0):
                        Clinks[i][j] = 1

            Csparse = Clinks
            SAC = Csparse + np.transpose(Csparse)
            ASAC = Csparse - np.transpose(Csparse)
            comm = np.subtract(np.matmul(SAC, ASAC).shape, np.matmul(ASAC, SAC).shape)
            a = np.matmul(SAC, ASAC).shape
            b = np.matmul(ASAC, SAC).shape
            comm = np.subtract(a,b)
            polys.append(np.concatenate(((np.poly(SAC)), (np.poly(ASAC)), (np.poly(comm))), axis=None))


        counter = tally(polys)
        distinct = len(counter) + 1
        print(Nsamples, " ", distinct)
    #AppendTo[plotdistinct, {i, distinct}]



if __name__ == "__main__":
    main()
