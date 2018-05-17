# -*- coding: utf-8 -*-

"""
Created on Thu Apr 20 07:50:00 2018

@author: Soma Turi

"""
# PLEASE SCROLL DOWN TO SET UP INPUT PARAMETERS AND PATHS
# what is needed to be set up:
# 1. paths to the input and output files
# 2. doTest: do you want to partition the input set into a test and training set to test the model? 
# 3. parameter: change the bestparam list variable, if given parameter combination is to be used, otherwise CV runs on training set 
# to determine it 



# IMPORTING modules and functions ---------------------------------------------------------------------

from math import log10, floor
from sklearn.neural_network import MLPRegressor
import csv
import numpy
from sklearn.model_selection import train_test_split
import pickle
from multiprocessing import Pool, RawArray
from scipy import stats
from numpy import linalg
import math
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools as AvT
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# FUNCTIONS used by the script ------------------------------------------------------------------------------------------------------------------------------
# the functions should work as follows:
# importdata imports the traindata from the smiles string set, the pairs the 
# testtable positives with their respective smiles string. It then converts the smiles 
# strings to bitvector arrays, both for the positive train, test and negative train, test sets


# giving values to four significant figures
def round_sig(x, sig=4):
    return round(x, sig-int(floor(log10(abs(x))))-1)


# IMPORT data, merge, split randomly into test and train+cv
# from two places, one is smiles paired up with results and ids, second is smiles paired up with ids and ids with results according to protein
    
def importdata(proteinname, proteinlocation, testtable, smiletable, doTest):
    # imports into list, deletes wrong ones
    if proteinlocation != '': 
        with open(proteinlocation,'rU') as f:
                reader = csv.reader(f, dialect=csv.excel_tab,delimiter="\t")
                d = list(reader) 
        smi = [item[0] for item in d]
        result = [item[2] for item in d]
    if smiletable != '':             
        with open(smiletable,'rU') as s:
            readert = csv.reader(s, dialect=csv.excel_tab,delimiter="\t")
            dt = list(readert) 
        smit = [item[0] for item in dt]
        idst = [item[1] for item in dt]
        basis = dict(zip(idst, smit))
    
    # gets the positive and negative ids for the proteins
    if testtable != '':
        f = numpy.genfromtxt(testtable, dtype='str')
        collumnnumber = int(numpy.where( f == proteinname)[1])
        posrownumber = list(numpy.where(f.T[collumnnumber] == '1')[0])
        negrownumber = list(numpy.where(f.T[collumnnumber] == '0')[0])
        posrowid = [j for i, j in list(enumerate(f.T[0])) if i in posrownumber]
        negrowid = [j for i, j in list(enumerate(f.T[0])) if i in negrownumber]
        posres = [1] * len(posrowid)
        negres = [0] * len(negrowid)
        possmiles = [0] * len(posrowid)
        negsmiles = [0] * len(negrowid)
        for i in range(0,len(posrowid)):
            if posrowid[i] in basis:
                possmiles[i] = basis[posrowid[i]]
            else: 
                print('mistake')
        for j in range(0,len(negrowid)):
            if negrowid[j] in basis:
                negsmiles[j] = basis[negrowid[j]]
            else:
                print('mistake')
        smi = smi + possmiles + negsmiles
        result = result + posres + negres
    
    posdata = [j for i, j in list(enumerate(smi)) if result[i] == '1']
    negdata = [j for i, j in list(enumerate(smi)) if result[i] == '0']
    numpy.random.shuffle(posdata)
    numpy.random.shuffle(posdata)
    numpy.random.shuffle(posdata)
    numpy.random.shuffle(posdata)
    numpy.random.shuffle(posdata)
    numpy.random.shuffle(negdata)
    numpy.random.shuffle(negdata)
    numpy.random.shuffle(negdata)
    numpy.random.shuffle(negdata)
    numpy.random.shuffle(negdata)
    if doTest == True: 
        trainpositives = posdata[0:int(len(posdata)*0.9)]
        trainnegatives = negdata[0:int(len(negdata)*0.9)]
        testpositives = posdata[int(len(posdata)*0.9):]
        testnegatives = negdata[int(len(negdata)*0.9):]
    if doTest == False: 
        trainpositives = posdata
        trainnegatives = negdata
        testpositives = []
        testnegatives = []
    return trainpositives, trainnegatives, testpositives, testnegatives


# CONVERT input of tuple - smiles and results to bitvectors of positives and negatives with the given parameters
    
def convertdata(positives, negatives, bitsize, radius):
    
    posmolecules = [Chem.MolFromSmiles(x) for x in positives]
    null = [i for i, item in enumerate(posmolecules) if item is None] 
    for i in sorted(null, reverse=True): 
        del posmolecules[i]

    # converts into morgan bitvectors
    
    Morgan6pos = [AllChem.GetMorganFingerprintAsBitVect(y,int(radius),nBits=int(bitsize)).ToBitString() for y in posmolecules]
    Avalonpos = [AvT.GetAvalonFP(y).ToBitString() for y in posmolecules]
    Combinedpos = [Morgan6pos[i] + y for i,y in list(enumerate(Avalonpos))]
    ACombinedpos =[numpy.array(list(map(int, x))) for x in Combinedpos]
    
    negmolecules = [Chem.MolFromSmiles(x) for x in negatives]
    null = [i for i, item in enumerate(negmolecules) if item is None] 
    for i in sorted(null, reverse=True): 
        del negmolecules[i]

    # converts into morgan bitvectors
    Morgan6neg = [AllChem.GetMorganFingerprintAsBitVect(y,int(radius),nBits=int(bitsize)).ToBitString() for y in negmolecules] 
    Avalonneg = [AvT.GetAvalonFP(y).ToBitString() for y in negmolecules]
    Combinedneg = [Morgan6neg[i] + y for i,y in list(enumerate(Avalonneg))]
    ACombinedneg =[numpy.array(list(map(int, x))) for x in Combinedneg]
    
     
    return ACombinedpos, ACombinedneg


# TRAIN the F and G function   
 
def functiontrain(Ftrain, Ftruth, Gtrain, Gtruth, Floc, Gloc):
    # impor train data
    with open(Ftrain,'rU') as f: 
        re= csv.reader(f,dialect=csv.excel_tab,delimiter=",")
        inputJ = list(re)
    with open(Ftruth,'rU') as f: 
        re= csv.reader(f,dialect=csv.excel_tab,delimiter=",")
        JJ = list(re)
    inputJ = numpy.array(inputJ).astype("float")
    JJ = numpy.array(JJ).astype("float")
    
    # training
    X_train, X_test, y_train, y_test = train_test_split(inputJ,JJ.ravel(), test_size=0.05, random_state=0)
    clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(9,7,5,),alpha=20,max_iter=500000)
    clf.fit(X_train,y_train.ravel())
    scoreF = clf.score(X_test,y_test)
    print('F function defined')
    filename = Floc
    pickle.dump(clf, open(filename, 'wb'))
 
    # open test data for G function training
    with open(Gtrain,'rU') as f:
        re= csv.reader(f,dialect=csv.excel_tab,delimiter=",")
        inputh = list(re)
    with open(Gtruth,'rU') as f: 
        re= csv.reader(f,dialect=csv.excel_tab,delimiter=",")
        hh = list(re)
    inputh = numpy.array(inputh).astype("float")
    hh = numpy.array(hh).astype("float")
    hh = hh[numpy.isfinite(inputh).all(axis=1)]
    inputh = inputh[numpy.isfinite(inputh).all(axis=1)]
        
    # training G
    X_train, X_test, y_train, y_test = train_test_split(inputh, hh.ravel(), test_size=0.2, random_state=0)
    # creates training and test set for learning G 
    clf1 = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(2,2,),activation='tanh',alpha=40)
    # creates neural network which will learn G - why tanh here? given dataset 
    # X learns G(x)
    clf1.fit(X_train,y_train.ravel())
    # trains neural network on dataset - for every example, create pred, and backprop to 
    # compute grads - make steps in changing parameters to reach to bet
    scoreG = clf1.score(X_test,y_test) 
    # tests it on test set, gives error metric
    print('G function defined')

    # testing G    
    filename = Gloc
    pickle.dump(clf1, open(filename, 'wb'))
    
    return clf, clf1, scoreF, scoreG


# TRAIN the model on a given training set, with given F and G and a given variance limit
    

def tester(Jp, Jn, hp, hn, poste, negte, plotloc, dataloc, index, proteinname, iscv):
    
    poste1 = numpy.matrix(poste)
    negte1 = numpy.matrix(negte)

    print('setting up testing data')
    post = numpy.matrix([j for i, j in enumerate(poste1.T.tolist()) if i not in index]).T
    negt = numpy.matrix([j for i, j in enumerate(negte1.T.tolist()) if i not in index]).T
    post = numpy.multiply(post, 2) -1
    negt = numpy.multiply(negt, 2) -1
        
 


    print('creating testing matrices')
    Epos_post = numpy.matrix(numpy.diag(post * numpy.matrix(Jp) * post.T)).T / 2 + post * numpy.matrix(hp).T
    Eneg_post = numpy.matrix(numpy.diag(negt * numpy.matrix(Jp) * negt.T)).T / 2 + negt * numpy.matrix(hp).T
    Epos_negt = numpy.matrix(numpy.diag(post * numpy.matrix(Jn) * post.T)).T / 2 + post * numpy.matrix(hn).T
    Eneg_negt = numpy.matrix(numpy.diag(negt * numpy.matrix(Jn) * negt.T)).T / 2 + negt * numpy.matrix(hn).T
       
# evaluating model on test set
    print('Scoring')
    Etresh = numpy.arange(-3,4, 0.0005)
    k = len(Etresh) / 1000
    sumfpt = numpy.zeros(len(Etresh))
    sumtpt = numpy.zeros(len(Etresh))
    print('Start')
        
    for i in range(0,len(Etresh)):
            
        sumfpt[i] = sum(1 for j in -(Eneg_post-Eneg_negt) / 10**3.2 if j < Etresh[i]) / len(Eneg_post)
        sumtpt[i] = sum(1 for k in -(Epos_post-Epos_negt) / 10**3.2 if k < Etresh[i]) / len(Epos_post)
        if i % 1000 == 0:
            z = i / 1000
            print(str(z) + ' thousand out of ' + str(k))
    
    #  evaluating on training set
    integral1 = numpy.trapz(sumtpt, x = sumfpt)
    print('i1 = ' + str(integral1))
    
    if iscv == False: 
       
        print('Doing plotting')
        plt.plot(sumfpt, sumtpt, label='test, i = ' + str(round_sig(integral1)), linestyle='--', marker='o')
        plt.legend()
        plt.xlabel('false positives')
        plt.ylabel('true positives')
        plt.title('Inverse Ising inference model testing for ' + proteinname)
#        plt.title('Model testing, integral = ' + str(integral1))
        plt.savefig(plotloc)
        plt.close()
        print('plotting ready')
        numpy.savetxt(dataloc, numpy.vstack((sumfpt, sumtpt)).T, delimiter = ',')
        
        
    
    return integral1
    
    
def modeller(Ffunction, Gfunction, postr, negtr, Jlocp, Jlocn, hlocp, hlocn, iscv, varlimit):
    
        # making matrix and eliminating random collumns
	
        postr1 = numpy.matrix(postr)
        negtr1 = numpy.matrix(negtr)
        
        standard_dev_pos = postr1.std(0)
        standard_dev_neg = negtr1.std(0)
        # list of indices, where small standard deviation
        index_pos = [i for i, j in enumerate(standard_dev_pos.A1) if j < varlimit]
        index_neg = [i for i, j in enumerate(standard_dev_neg.A1) if j < varlimit]
        index = numpy.unique(index_pos + index_neg)
        # taking only thoose collumns which has considerable st. dev
        pos = numpy.matrix([j for i, j in enumerate(postr1.T.tolist()) if i not in index]).T
        neg = numpy.matrix([j for i, j in enumerate(negtr1.T.tolist()) if i not in index]).T
        pos = numpy.multiply(pos, 2) - 1
        neg = numpy.multiply(neg, 2) -1
       # creates inputs
        mupos = numpy.matrix(pos.mean(0))
        sigmapos = numpy.matrix(pos.std(0))
        zpos = stats.zscore(pos)
        muneg = numpy.matrix(neg.mean(0))
        sigmaneg = numpy.matrix(neg.std(0))
        numpy.seterr(invalid="raise")
        zneg = stats.zscore(neg)
        # giving number of positive, negative elements (rows) as scalar integer
        Npos = zpos.shape[0]
        Nneg = zneg.shape[0]          
        # creates the positive inputs
        # creating an M x M correlation matrix, where M is number of remained features
        Cpos = numpy.divide((zpos.T * zpos), Npos) 
        # creates an array of dpos eigenvalues and a matrix of respective vpos eigenvectors (MxM)
        dpos, vpos = linalg.eigh(Cpos)
        vpos = numpy.matrix(vpos)
        y = zpos.shape[1] / Npos
        # M x 1 collumn matrix of eigenvalues
        dpos = numpy.matrix([list(dpos)]).T
        # zero the wrong eigenvalues, keep mX1
        dposclip = numpy.multiply((dpos > (1 + math.sqrt(y)) **2), dpos)
        # creates an mxm array of dposclip, an mxm matrix with dposclip as diagonal elements
        Cposclip = vpos*numpy.diag(dposclip.A1)*vpos.T + numpy.eye(zpos.shape[1]) * numpy.asscalar(numpy.array(dpos[[i for i, j in enumerate(dpos) if j < (1 + math.sqrt(y)) ** 2]].sum(0))) / zpos.shape[1]    
        # creates an array with mxm , only diagonal elements, mxm 
        Cmod_pos = numpy.diag(sigmapos.A1) * Cposclip * numpy.diag(sigmapos.A1)
        Cmod_pos = numpy.matrix(Cmod_pos)
        J_pos = -numpy.linalg.inv(Cmod_pos)
        # clears diagonal
        J_pos = J_pos - numpy.diag(numpy.diag(J_pos))
        
        # creates negative inputs
        
        Cneg = numpy.divide((zneg.T * zneg), Nneg)
        dneg, vneg = linalg.eigh(Cneg)
        vneg = numpy.matrix(vneg)
        y = zneg.shape[1] / Nneg
        dneg = numpy.matrix([list(dneg)]).T
        dnegclip = numpy.multiply((dneg > (1 + math.sqrt(y)) ** 2), dneg)
        Cnegclip = vneg*numpy.diag(dnegclip.A1)*vneg.T + numpy.eye(zneg.shape[1]) * numpy.asscalar(numpy.array(dneg[[i for i, j in enumerate(dneg) if j < (1 + math.sqrt(y)) ** 2]].sum(0))) / zneg.shape[1]
        Cmod_neg = numpy.diag(sigmaneg.A1) * Cnegclip * numpy.diag(sigmaneg.A1)
        Cmod_neg = numpy.matrix(Cmod_neg)
        J_neg = -numpy.linalg.inv(Cmod_neg)
        J_neg = J_neg - numpy.diag(numpy.diag(J_neg))

        # put the input values into collumn vector form
        B = numpy.tril(numpy.ones((Cmod_pos.shape[0], Cmod_pos.shape[0])), -1)
        parCor_pos = numpy.array(J_pos.T[numpy.where(B == 1)].T)
        TotCor_pos = numpy.array(Cmod_pos.T[numpy.where(B == 1)].T)      
        mux_pos = numpy.matlib.repmat(mupos, Cpos.shape[0], 1)
        muy_pos = numpy.matlib.repmat(mupos.T, 1, Cpos.shape[0])
        mux_pos = numpy.array(mux_pos.T[numpy.where(B == 1)].T)
        muy_pos = numpy.array(muy_pos.T[numpy.where(B == 1)].T)
        
        parCor_neg = numpy.array(J_neg.T[numpy.where(B== 1)].T)
        TotCor_neg = numpy.array(Cmod_neg.T[numpy.where(B == 1)].T)
        mux_neg = numpy.matlib.repmat(muneg, Cneg.shape[0], 1)
        muy_neg = numpy.matlib.repmat(muneg.T, 1, Cneg.shape[0])
        mux_neg = numpy.array(mux_neg.T[numpy.where(B == 1)].T)
        muy_neg = numpy.array(muy_neg.T[numpy.where(B == 1)].T)
        print('starting lists')
        
        # creates a list - with list elements, each list element has four value, just like the
        # input of the F function, I hope in rigth order, the right input
        inputJ_negf = []
        inputJ_posf = []
        for i in range(0,len(parCor_pos)):
            inputJ_negf.append([numpy.asscalar(parCor_neg[i]), numpy.asscalar(TotCor_neg[i]), numpy.asscalar(mux_neg[i]), numpy.asscalar(muy_neg[i])])
            inputJ_posf.append([numpy.asscalar(parCor_pos[i]), numpy.asscalar(TotCor_pos[i]), numpy.asscalar(mux_pos[i]), numpy.asscalar(muy_pos[i])])
        print('lists made') 
        
        
        # for every key, assign in the output to the key the list
        # containing the info for the negative and positive F function
        pospredictj = Ffunction.predict(inputJ_posf)
        negpredictj = Ffunction.predict(inputJ_negf)
        print('predictions made')
       
        ## builds back matrix
        # creates a tuple of tubhe nonozero, i.e. 1 elements, in a good order
        Jpospredict = numpy.zeros(B.shape)
        Jnegpredict = numpy.zeros(B.shape)
        Intermedpos = list(zip(numpy.nonzero(B)[0], numpy.nonzero(B)[1]))
        # every matrix coordinate specified by intermedpos, changes the value to the one imported
        for i, j in list(enumerate(pospredictj)):
            Jpospredict[Intermedpos[i]] = j
        Jpospredict = Jpospredict + Jpospredict.T
        print('pospredict made')
        
        Intermedneg = list(zip(numpy.nonzero(B)[0], numpy.nonzero(B)[1]))
        for i, j in list(enumerate(negpredictj)):
            Jnegpredict[Intermedneg[i]] = j
        Jnegpredict = Jnegpredict + Jnegpredict.T
        print('negpredict made')
#        
        ## inpuit for G
        hmupos = numpy.arctanh(mupos).T
        hmuneg = numpy.arctanh(muneg).T
        print(hmupos.shape)
        secneg = numpy.matrix(numpy.diag(-numpy.linalg.inv(numpy.matrix(Cmod_neg)))).T
        secpos = numpy.matrix(numpy.diag(-numpy.linalg.inv(numpy.matrix(Cmod_pos)))).T
        print(secneg.shape)
        thirdneg = numpy.matrix((Cmod_neg - numpy.diag(numpy.diag(Cmod_neg)))) * numpy.matrix(muneg).T
        thirdpos = numpy.matrix((Cmod_pos - numpy.diag(numpy.diag(Cmod_pos)))) * numpy.matrix(mupos).T
        print(thirdpos.shape)
        fourthneg = numpy.matrix(Jnegpredict) * numpy.matrix(muneg).T
        fourthpos = numpy.matrix(Jpospredict) * numpy.matrix(mupos).T
        print(fourthpos.shape)
        
        # creating input list
        inputh_negf = []
        inputh_posf = []
        for i in range(0,len(hmupos)):
            inputh_negf.append([numpy.asscalar(hmuneg[i]), numpy.asscalar(secneg[i]), numpy.asscalar(thirdneg[i]), numpy.asscalar(fourthneg[i])])
            inputh_posf.append([numpy.asscalar(hmupos[i]), numpy.asscalar(secpos[i]), numpy.asscalar(thirdpos[i]), numpy.asscalar(fourthpos[i])])

        
        ## predicting h
        pospredict_h = Gfunction.predict(inputh_posf)
        negpredict_h = Gfunction.predict(inputh_negf)
        
        if iscv == False: 
            fileObject = open(Jlocp,'wb')
            pickle.dump(Jpospredict,fileObject)   
            fileObject.close()
    
            fileObject = open(Jlocn,'wb')
            pickle.dump(Jnegpredict,fileObject)   
            fileObject.close()
    
            fileObject = open(hlocp,'wb')
            pickle.dump(pospredict_h,fileObject)   
            fileObject.close()
            
            fileObject = open(hlocn,'wb')
            pickle.dump(negpredict_h,fileObject)   
            fileObject.close()

        

        return Jpospredict, Jnegpredict, pospredict_h, negpredict_h, index
# test the model on a given test set, with a given learned J, h and indeces of removed rows 
        # creating matrices for testing
        


# makes 9/10-fold crossvalidation with a given set of parameters, using shared input arrays and F & G

def crossvalidation(Ffunction, Gfunction, r, b, v, istest):
    
    if istest == True:
         n = 9
    if istest == False:
         n = 10

    PT_np = numpy.frombuffer(var_dict['PT']).reshape(var_dict['pshape'])
    TN_np = numpy.frombuffer(var_dict['TN']).reshape(var_dict['nshape'])

    poslength = len(PT_np)
    neglength = len(TN_np)
    integrals = numpy.ones((n,1))
    for k in range(1, n+1):
        
        if k < n:
            poscv = PT_np[int((k-1)*1/n*poslength):int(k*1/n*poslength)]
            negcv = TN_np[int((k-1)*1/n*neglength):int(k*1/n*neglength)]
            print('cv created ' + str(k) + ' for variance limit ' + str(v))
            postrcv = numpy.concatenate((PT_np[:int((k-1)*1/n*poslength)], PT_np[int(k*1/n*poslength):]),0)
            negtrcv = numpy.concatenate((TN_np[:int((k-1)*1/n*neglength)], TN_np[int(k*1/n*neglength):]),0)
            Jp, Jn, hp, hn, i= modeller(Ffunction, Gfunction, postrcv, negtrcv, '', '', '', '', True, v)
            integrals[k-1]= tester(Jp, Jn, hp, hn, poscv, negcv, '', '', i, '', True)
                
            
        if k == n:
            poscv = PT_np[int((k-1)*1/n*poslength):]
            negcv = TN_np[int((k-1)*1/n*neglength):]
            print('cv created ' + str(k) + ' ')
            postrcv = PT_np[:int((k-1)*1/n*poslength)]
            negtrcv = TN_np[:int((k-1)*1/n*neglength)]
            Jp, Jn, hp, hn, i= modeller(Ffunction, Gfunction, postrcv, negtrcv, '', '', '', '', True, v)
            integrals[k-1]= tester(Jp, Jn, hp, hn, poscv, negcv, '', '', i,  '', True)
                        
    
    average = numpy.mean(integrals, 0)
    std = numpy.std(integrals, 0)
    result = [r, b, v, numpy.asscalar(average), numpy.asscalar(std)]
    
    return result 
                

# defines dictionary to store shared array

var_dict = {}
  
def init_worker(PT, pshape, TN, nshape):
     var_dict['PT'] = PT
     var_dict['TN'] = TN
     var_dict['pshape'] = pshape
     var_dict['nshape'] = nshape




# -------------------------------------------------------------------------------------------------------------------------------------------------


#MAIN PART OF THE SCRIPT - PLEASE EDIT ACCORDINGLY-------------------------------------------------------------------------------------------------
if __name__ == '__main__':


# SPECIFYING PATH TO FILES - please always give full paths, compatible with the operation system used, in ''-s------------------------------------

    # path to F and G functions
    Fpath = '/home/st716/script/script/savednetwork/finalized_modelf3.sav'
    Gpath = '/home/st716/script/script/savednetwork/finalized_modelg3.sav'
    Gtrain, Ftrain, Gtruth, Ftruth = [''] * 4
    # if multiple runs for the same protein are done, the number of the run - to avoid overwriting existing files - leave '' if single run
    number = '1'
    # path to protein smiles - leaves '' if not used
    protsmal = 'nr-ahr'
    protpath = '/home/st716/script/script/protdata/' + protsmal + '.smiles'
    
    # path to testtable and smiletable - leave '' if not used
    protname = 'NR-AhR'
    ttable = '/home/st716/script/script/testdata/tox21_10k_challenge_score.txt'
    stable = '/home/st716/script/script/testdata/tox21_10k_challenge_score.smiles'
    
    # path to output J, h - adivsable to change only folder 
    Jppath = '/home/st716/script/script/modeldescriptors/' + protsmal + 'jp' + number + '.sav'
    Jnpath = '/home/st716/script/script/modeldescriptors/' + protsmal + 'jn' + number + '.sav'
    hppath = '/home/st716/script/script/modeldescriptors/' + protsmal + 'hp' + number + '.sav'
    hnpath = '/home/st716/script/script/modeldescriptors/' + protsmal + 'hn' + number + '.sav'
    
    # path to output fp-tp if test set - advisable to change only folder 
    datapath = '/home/st716/script/script/plotdatas/' + protsmal + number + '.txt'
    plotpath = '/home/st716/script/script/dataplots/' + protsmal + number + '.png'
    # path to output parameter combination - advisable to change only folder 
    useddatapath = '/home/st716/script/script/useddatas/' + protsmal + number + '.txt'

# ---------------------------------------------------------------------------------------------------------------------------------------------
# IMPLEMENTING the modelling-------------------------------------------------------------------------------------------------------------------

    # importing the trained functions      
    Ffunction = pickle.load(open(Fpath, 'rb'))
    Gfunction = pickle.load(open(Gpath, 'rb'))
    
    # importing the training data (splitting the data into train and test set, if needed)
    doTest = True  # if False, give 1, if true 0.9 to importdata
    trainpos, trainneg, testpos, testneg = importdata(protname, protpath, ttable, stable, doTest)
        
    # create iteration field, please edit accordingly 
    radii = [3,4,5,6,7]
    varlimits = [0.20, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
    bitsizes = [330, 440, 512, 652, 768, 862, 1024, 1330, 1536, 1740, 2048]   
    averages = numpy.zeros((len(radii), len(bitsizes), len(varlimits)))
    
    # parameters used for training the modell on the whole set and the testing - change this to [radius, bitsize, varlimit] to skip CV
    bestparam = [] 
    
    # if [] = params, then undergo here CV, resulting in the best parameter combination - using the full training set or the 90% of the trainig set, if test set was wanted above
    if bestparam == []:
    
        for q,r in list(enumerate(radii)): 
        
                
            for i,l in list(enumerate(bitsizes)):
                
                print('import for bitsize ' + str(l) + ' and radius ' + str(r))
                postra, negtra = convertdata(trainpos, trainneg, l, r)
                postra = numpy.array(postra)
                negtra = numpy.array(negtra)
                pshape = postra.shape
                nshape = negtra.shape
                PT = RawArray('f', pshape[0]*pshape[1]*2)    
                TN = RawArray('f', nshape[0]*nshape[1]*2)
                PT_np = numpy.frombuffer(PT).reshape(pshape)
                TN_np = numpy.frombuffer(TN).reshape(nshape)
                numpy.copyto(PT_np, postra)
                numpy.copyto(TN_np, negtra)
    
                with Pool(processes=10, initializer=init_worker, initargs=(PT, pshape, TN, nshape)) as pool:  
                     results = pool.starmap(crossvalidation, [(Ffunction, Gfunction, r, l, s, doTest) for s in varlimits])     
                for t,z in list(enumerate(results)):
                     averages[q][i][t] = z[3]
        
    
        maxcoords = numpy.unravel_index(averages.argmax(), averages.shape)
        bestparam = [radii[maxcoords[0]], bitsizes[maxcoords[1]], varlimits[maxcoords[2]]]
        useddata = copy.copy(bestparam)
        useddata.append(numpy.amax(averages))
        numpy.savetxt(useddatapath, useddata, delimiter=',')
        
    
    #  converting data to best parameter combination, train the model on full training set, with best parameter combination, pickling out J, h 
    
    postrain, negtrain = convertdata(trainpos, trainneg, bestparam[1], bestparam[0])
    Jp, Jn, hp, hn, i = modeller(Ffunction, Gfunction, postrain, negtrain, Jppath, Jnpath, hppath, hnpath, False, bestparam[2])
    
    # if there was a test set, convert test set to best parameters, evaluate and score on test
    if doTest == True:
        postest, negtest = convertdata(testpos, testneg, bestparam[1], bestparam[0])
        testintegral = tester(Jp, Jn, hp, hn, postest, negtest, plotpath, datapath, i, protname, False) 