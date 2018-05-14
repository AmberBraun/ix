"""
Author: Raja Harsha Chinta
Title: Alternating Least Square Implementation in Apache Spark for Product Recommendation
Sample Call: spark-submit als_2.py AZ_P/input AmazonProductReco.dat 10 4

Instructions:

	Step 1: Open Hortonworks Spark Virtual box/ spark environment with atleast Spark 1.2.1 and Python 2.7.7 installed.
	Step 2: If required install numpy, scipy libraries.
	Step 3: Execute the command "export SPARK_HOME=/usr/hdp/2.2.4.2-2/spark" in unix server.
	Step 4: Transfer the code file AmazonALS.py, ratings.dat, products.dat, users.dat to unix server.
	Step 5: Create a directory in HDFS like AZ/input and trandfer the .dat files to input HDFS directory created.
	Step 6: To run the recommendation program execute: spark-submit AmazonALS.py <InputDirectory> <OutputFileName> <Iterations> <Partitions>
	Step 7: If the number of iterations are more than 10, the program likely takes more than 15 minutes.
	Step 8: An output files is created in the same directory with userID,recommendedProduct,predictedRating

"""
from __future__ import print_function
# import sys
# import datetime
import numpy as np
# from scipy.linalg import solve
import time
# from math import sqrt
# from operator import add
# from datetime import datetime
from numpy.random import rand
from numpy import matrix
from multiprocessing import Process,Pool

# from os.path import join, isfile, dirname

# def loadRatings(ratingsFile):
#     """
#     Load ratings from file.
#     """
#     if not isfile(ratingsFile):
#         print ("File %s does not exist." % ratingsFile)
#         sys.exit(1)
#     f = open(ratingsFile, 'r')
#     ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
#     f.close()
#     if not ratings:
#         print ("No ratings provided.")
#         sys.exit(1)
#     else:
#         return ratings

def parseRating(line):
    """
    Parses a rating record in productLens format userId,productId,rating,timestamp .
    """
    fields = line.strip().split(",")
    return int(fields[0]), int(fields[1]), float(fields[2])

def parseProduct(line):
    """
    Parses a product record in productLens format productId,productTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]

def parseUser(line):
    """
    Parses a User record in productLens format productId,productTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]    

def computeRMSE(R, us, vs):
    """
    compute Root mean square error value
    """
    diff = R - us[:,:D] * vs[:,:D].T
    rmse_val = np.sqrt(np.sum(np.power(diff, 2))/(U * V))
    return rmse_val

def updateUV(i, uv, r):
    """
    Calculate updated values of U,V
    """
    uu = uv.shape[0]
    ff = uv.shape[1]
    xtx = uv.T * uv
    xty = uv.T * r[i, :].T

    for j in range(ff):
        xtx[j, j] += lambdas * uu
    
    updated_val = np.linalg.solve(xtx, xty)
    return updated_val

def updateUorV(v,D):##矩阵
    for i in range(v.shape[0]):
        v[i,D:] = np.array(matrix(v[i,:D]).T * matrix(v[i,:D])).flatten()
    return v
    # vv = v.shape[0]
    # ff = v.shape[1]
    # xtx = v.T * v
    # xty = v.T * r[i, :].T
    #
    # for j in range(ff):
    #xtx[j, j] += lambdas * vv
    #
    # updated_U = np.linalg.solve(xtx, xty)

# def pull(v,u,D,L,r):
#     col = r.shape[1]
#     row = r.shape[0]
#     u[:, 0:int(D / L)] = r * v[:,0:int(D / L)]###vec
#     count=False
#     a = np.zeros((1, int(D * D / L)))
#     for i in range(row):
#         for j in range(col):
#             if (r[i,j] != 0):
#                 count=True
#                 a = a + v[j, int(D / L):]
#         if(count==True):
#             u[i, int(D / L):] = a.copy()
#             count=False
#             a = np.zeros((1, int(D * D / L)))
#     return u
def pull(vvec,vmat,uvec,umat,D,L,r):
    col = r.shape[1]
    row = r.shape[0]
    uvec[:, 0:] = r * vvec[:,0:]###vec
    count=False
    a = np.zeros((1, int(D * D / L)))
    for i in range(row):
        for j in range(col):
            if (r[i,j] != 0):
                count=True
                a = a + vmat[j, :]
        if(count==True):
            umat[i, :] = a.copy()
            count=False
            a = np.zeros((1, int(D * D / L)))
        # c = np.concatenate((uvec.flatten(), umat.flatten()), axis=1)
    return (uvec,umat)

# def push(u,v,D,L,r):##分区好的u,v
#     col=r.shape[1]
#     row=r.shape[0]
#     #print(r.shape)
#     v[:,0:int(D/L)] =r.T*u[:,0:int(D/L)]
#     count=False
#     a = np.zeros((1, int(D * D / L)))
#     for i in range(col):
#         for j in range(row):
#             #print(r[j,i])
#             if(r[j,i]!=0.):
#                 count=True
#                 a=a+u[j,int(D / L):]
#         #print(a)
#         if(count==True):
#
#             v[i,int(D / L):]=a.copy()
#             count=False
#             a = np.zeros((1, int(D * D / L)))
#     return v
def push(uvec,umat,vvec,vmat,D,L,r):##分区好的u,v
    col=r.shape[1]
    row=r.shape[0]
    #print(r.shape)
    vvec[:,0:] =r.T*uvec[:,0:]
    count=False
    a = np.zeros((1, int(D * D / L)))
    for i in range(col):
        for j in range(row):
            #print(r[j,i])
            if(r[j,i]!=0.):
                count=True
                a=a+umat[j,:]
        #print(a)
        if(count==True):

            vmat[i,0:]=a.copy()
            count=False
            a = np.zeros((1, int(D * D / L)))
    # c = np.concatenate((uvec.flatten(), umat.flatten()), axis=1)
    # b=[]
    # for i in range(c.size):
    #     b.append(c[0,i])
    return (vvec,vmat)
def updateUorVF3(D,v,L,vv):
    #vv= v.shape[0]
    for i in range(v.shape[0]):
        #print('当前：',i)
        #if i==126:print(matrix(v[i,D:]).reshape((D,D)))
        xtx=matrix(v[i, int(D / L):]).reshape((int(D / L), int(D / L)))
        for j in range(int(D / L)):
            xtx[j,j]+=lambdas*vv
        v[i,:int(D/L)] = matrix(np.linalg.solve(xtx,v[i,0:D].T)).T
    #print('shape:',matrix(updated).shape)
    return v
# def updateV(i, u, r):
#
#     uu = u.shape[0]
#     ff = u.shape[1]
#     xtx = u.T * u
#     xty = u.T * r[i, :].T
#
#     for j in range(ff):
#         xtx[j, j] += lambdas * uu
#
#     updated_V = np.linalg.solve(xtx, xty)
#     return updated_V
        
if __name__ == "__main__":
    print('输入L：')
    L = int(input())
    print('输入D：')
    D = int(input())
    lastrmse_val=10000
    # if (len(sys.argv) != 2):
    #      print ("请输入正确的参数")
    #      sys.exit(1)
    #partitioned=sys.argv[0]
    partitioned=False
    # parameters are declared
    lambdas = 0.1
    np.random.seed(20)
    #hdfs_src_dir = sys.argv[0]
    iterations = 6
    #partitions = 4
    start_time = time.time()
    #outputfile = sys.argv[2]

    # AppName, memory is set to SparkContext
    # conf = SparkConf().setAppName("AmazonALS").set("spark.executor.memory", "2g")
    # sc = SparkContext(conf=conf)

    # ratings is an RDD of (timestamp, (userId, productId, rating))
    #ratings = sc.textFile(join(hdfs_src_dir, "ratings.dat")).map(parseRating)
    f=open('E:\\ratings.txt','r')
    rating=f.readlines()
    ratings=[]
    for row in rating:
        ratings.append(parseRating(row))
    # products is an RDD of (productId, productTitle)
    #products = sc.textFile(join(hdfs_src_dir, "products.dat")).map(parseProduct)
    product= open('E:\\products.txt','r').readlines()
    products=[]
    for row in product:
        products.append(parseProduct(row))
    # users is an RDD of (userID, userName)
    #users = sc.textFile(join(hdfs_src_dir, "users.dat")).map(parseUser)
    users=[]
    user= open('E:\\users.txt','r').readlines()
    for row in user:
        users.append(parseUser(row))
    #r_list = ratings
    r_array = np.array(ratings)

    numRatings = r_array.shape[0]
    U = int(max(r_array[:,0]))
    V = int(max(r_array[:,1]))
    Upar=int(U/2)
    Vpar=int(V/2)
    print(U,V,numRatings)

    Z = np.zeros((U,V))
    R = np.matrix(Z)
    for i in range(numRatings):
        r_local = ratings[i]
        # print(r_local[0])
        # print()
        R[(r_local[0]-1),(r_local[1]-1)] = r_local[2]
    #print(R[:50,:50])
  
    us =  matrix(rand(Upar, D+D*D))
    #usb = sc.broadcast(us)
    vs =  matrix(rand(Vpar, D+D*D))
    # p=[]
    # for i in range(L):
    # p1 = Process()
    # p2 = Process()
    #for i in range(F):#层数
    # (pipe1, pipe2) = Pipe()

    for i in range(iterations):
        #vs = list(map(lambda x: updateUV(x, us, R.T), range(V)))
        #print(us[0,:])
        us = updateUorV(us, D)
        #out_put=pool.map(lambda x:push(matrix(us[:,int(x*D/L):int((x+1)*D/L)]),matrix(us[:,int(D+x*D*D/L):int(D+(x+1)*D*D/L)]),matrix(vs[:,int(x*D/L):int((x+1)*D/L)]),matrix(vs[:,int(D+x*D*D/L):int(D+(x+1)*D*D/L)]),D,L,R),range(L))
        # for row in range(len(out_put)):
        #     vs[:,row*int(D/L):(row+1)*int(D/L)]=matrix(out_put[row][0]).reshape((V,int(D/L)))
        #     vs[:, D+row * int(D*D / L):D+(row + 1) * int(D*D / L)] = matrix(out_put[row][1]).reshape((V,int(D*D/L)))
        # for row in range(len(out_put)):
        #
        #     vs[:,row*int(D/L):(row+1)*int(D/L)]=matrix(out_put[row][0:V*int(D/L)]).reshape((V,int(D/L)))
        #     vs[:, D + row * int(D * D / L):D + (row + 1) * int(D * D / L)]=matrix(out_put[row][V*int(D/L):]).reshape((V,int(D*D/L)))
        output=[]
        pool = Pool(processes=L)
        for j in range(L):
            output.append(pool.apply_async(push,args=(matrix(us[:,int(j*D/L):int((j+1)*D/L)]),matrix(us[:,int(D+j*D*D/L):int(D+(j+1)*D*D/L)]),matrix(vs[:,int(j*D/L):int((j+1)*D/L)]),matrix(vs[:,int(D+j*D*D/L):int(D+(j+1)*D*D/L)]),D,L,R[:Upar,:Vpar])))
        #vs = push(us, vs, D, L, R)
        pool.close()
        pool.join()
        pool.terminate()
        m=0
        for res in output:
            ress=res.get()
            vs[:, m * int(D / L):(m + 1) * int(D / L)]=matrix(ress[0])
            vs[:, D + m * int(D * D / L):D + (m + 1) * int(D * D / L)]=matrix(ress[1])
            m=m+1
        vs = matrix(updateUorVF3(D, vs, 1, us.shape[0]))
        vs = updateUorV(vs, D)
        #us = pull(vs, us, D, L, R)
        # out_putput=list(pool.map(lambda x:pull(matrix(vs[:,int(x*D/L):int((x+1)*D/L)]),matrix(vs[:,int(D+x*D*D/L):int(D+(x+1)*D*D/L)]),matrix(us[:,int(x*D/L):int((x+1)*D/L)]),matrix(us[:,int(D+x*D*D/L):int(D+(x+1)*D*D/L)]),D,L,R),range(L)))
        # for row in range(len(out_putput)):
        #     us[:,row*int(D/L):(row+1)*int(D/L)]=matrix(out_putput[row][0]).reshape((U,int(D/L)))
        #     us[:, D+row * int(D*D / L):D+(row + 1) * int(D*D / L)] = matrix(out_putput[row][1]).reshape((U,int(D*D/L)))
        output=[]
        m=0
        pool = Pool(processes=L)
        for j in range(L):
            output.append(pool.apply_async(pull,args=(matrix(vs[:,int(j*D/L):int((j+1)*D/L)]),matrix(vs[:,int(D+j*D*D/L):int(D+(j+1)*D*D/L)]),matrix(us[:,int(j*D/L):int((j+1)*D/L)]),matrix(us[:,int(D+j*D*D/L):int(D+(j+1)*D*D/L)]),D,L,R[:Upar,:Vpar])))
        pool.close()
        pool.join()
        pool.terminate()
        for res in output:
            ress=res.get()
            us[:, m * int(D / L):(m + 1) * int(D / L)] = matrix(ress[0])
            us[:, D + m * int(D * D / L):D + (m + 1) * int(D * D / L)] = matrix(ress[1])
            m = m + 1
        us = matrix(updateUorVF3(D, us, 1, vs.shape[0]))
        #print(us[0,:])
        #vs = matrix(np.array(vs)[:, :, 0])
        # print(vs.shape)
        # print(us.shape)
        #print(vs[126,:])

        #print(vs[126, D:])

        #us = list(map(lambda x: updateUV(x, vs, R),range(U)))
        #us = matrix(np.array(us)[:, :, 0])
        #usb = sc.broadcast(us)


        #vsb = sc.broadcast(vs)

        rmse_val = computeRMSE(R[:Upar,:Vpar], us, vs)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % rmse_val)
        if lastrmse_val-rmse_val<=0.0001:
            print('以拟合')
            break
        else:
            lastrmse_val=rmse_val
    reco = us[:,:D]*vs[:,:D].T
            
    end_time = time.time()
    total_time  = end_time-start_time
    # total_t = divmod(total_time.days * 86400 + total_time.seconds, 60)

    print ("---------------------------------------------------------------------")
    print ("User-Product Recommendation: Predicted User-Ratings Matrix is created") 
    print ("---------------------------------------------------------------------")
    print("Total Minutes and Seconds: " ,total_time)

    l_prod = products
    # l_users = users.collect()
    # URatings = []
    # preURatings = []
    if(partitioned==False):
        output = open("output_Reco.dat",'w')
    else:
        output = open("3Doutput_eco.dat"+str(L)+'+'+str(D),'w')
    print("File writing started")

    for i in range(Upar):
        for j in range(Vpar):
            pRating = reco[i,j]
            aRating = R[i,j]
    #resorted = [sorted(x, reverse=True) for x in reco]
            if((aRating==0 and pRating>3.5)):
                output.write(str(i)+","+l_prod[j][1]+","+str(pRating)+"\n")
    
    output.close()

    """
   	     # Calcuate Average rating of users before and after prediction for testing purpose

             # if(aRating!=0):
	         # URatings.append(aRating)
		 # preURatings.append(pRating)

    # avgURating = float(sum(URatings))/float(len(uRatings))
    # avgPRating = float(sum(preURatings))/float(len(preURatings))

    # print ("Avg User Rating: ", avgURating)
    # print ("Avg Predicted User Rating: ", avgPRating )
    """
    

    # end_time = datetime.now()
    # total_time  = start_time - end_time
    # total_t = divmod(total_time.days * 86400 + total_time.seconds, 60)
    #
    # print ("-------------------------------------------------")
    # print ("User-Product Recommendation: File Write Completed")
    # print ("-------------------------------------------------")
    # print("Total Minutes and Seconds: " + str(total_t))

