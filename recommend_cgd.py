# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 11:24:14 2016

@author: Chen Guodan
"""
#class Static:
#    StaticVar = 5
#    def varfunc(self):
#        self.StaticVar += 1
#        print self.StaticVar
#
# a = Static()
#for i in range(3):
#    a.varfunc()
import numpy as np
import pandas as pd
class prob_spreading():
    def __init__(self, matrix,energy_item,num_user, num_item, train, validation, **params):
        self.matrix = matrix
        self.num_user = num_user
        self.num_item = num_item
        self.energy_item = energy_item
        self.degree_user = np.sum(self.matrix,axis=1)#column_sum axis = 0,row_sum axis=1
        self.degree_item = np.sum(self.matrix,axis=0)
        self.energy_item_new = np.zeros(num_item) 
 
        #np.zeors(num_item)
#sum_each_item average to user who used this items
    def item_to_user(self):
        mid = np.divide(self.energy_item ,self.degree_item)
        result = np.dot(self.matrix,mid)
        return result
    
    def user_to_item(self):
        energy_from_user = self.item_to_user()
        mid = np.divide(energy_from_user, self.degree_user)
        ma_t = self.matrix.transpose()
        result = np.dot(ma_t,mid)
        return result
    
    def weight(self):
        mid1 = np.divide(1,self.energy_item)
        ma_t = self.matrix.transpose()
        m1 = self.matrix/self.degree_user[:,None]#row of matrix is divided by the element of vector
        mid2 = np.dot(ma_t,m1)
        re = np.dot(mid1,mid2)
        return re
        
    def similarity_p(self,d=1.9):
        a= np.power(self.degree_item,d)
        s1 = np.transpose(self.matrix)/a[:,None]
        s2= np.dot(self.matrix,s1)
        sim = s2/self.degree_user # colums divide by element of vector
        return sim

        
        
    


        
#基于用户的协同过滤算法
class user_based_fl(prob_spreading):
    def __init__(self, matrix,energy_item,num_user, num_item, train, validation, **params):
        super(prob_spreading, self).__init__()
    
    def similarity(self):
        s1 = np.dot(self.matrix,np.transpose(self.matrix))
        s2 = np.dot(self.degree_user.transporse(),self.degree_user)
        s2 = np.sqrt(s2)
        sim = np.divide(s1,s2)
        return sim
        
    def predict_score(self):
        sim = self.similarity()
        s1 = np.sum(sim,axis=1) #row_sum here
        s1 = s1 - 1
        mid = np.dot(sim.transpose() - np.ones(self.matrix.shape), self.matrix)
        score = mid/s1[:,None]
        return score
        
    def recommend(self,k=5):
        score = self.predict_score()
        score[self.martrix==1]=0#set score(self.matrix==0) to 0
        ind = np.argsort(score, axis=1)#row sort from smallest to largest;list or oned_dim array,reserved:[::-1]
        ind_top = np.fliplr(ind)# resvered matrix by row
        ind_top5 = ind_top[:,:5]
        return ind_top5
        
        
    
#item based simarility
#点击率 add the click through rate to item_simarility x>=0
class item_based_fl():
    def __init__(self, matrix,energy_item,num_user, num_item, train, validation, click_through_rate,**params):
        super(prob_spreading, self).__init__()
        self.CRT = click_through_rate
    
    def item_simarility(self, x):
        
        
        
    def CTR_item_simarility(self, x):
        sim = self.item_similarity()
        mid1 = sim*(x+self.CRT)
        s1 = np.sum(sim,axis=1) #row_sum here
        s1 = s1 - 1
        mid = x + np.dot(sim.transpose() - np.ones(self.CRT.shape), self.CRT.shape)
        mid2 = mid/s1[:,None]# row divided     

class FTRL_Proximal():
    def __init__(self):
    
    
    def (alpha_1,alpha_2,lambda_1,lambda_2):
        
        
        
        
        
        
        
        
    
       
        
        
        
        

        
          
        
        
        

            
        
                