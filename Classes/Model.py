# adapted from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures.git
# %%
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time

from IPython import embed

from itertools import combinations_with_replacement as comb

class Model():

    def __init__(self, integration, rule, height, deg, derivative = False):
        # Integration function that creates integration({tree: values}) returns {I[tree]: I[values]}
        self.I = integration 
        self.deg = deg # maximum degree of the created trees
        self.H = height # maximum height of the trees 
        self.R = rule  # Rule involving several extra trees and widths
        self.models = None
        self.size = 0 # number of realizations of models
        self.derivative = derivative # True if derivatives are present in the model. At the moment only differentiation order <= 1 are allowed
        
    def return_model(self, i): # returns the value of the i-th model
        if type(self.models) == list:
            return self.models[i-1]
        return self.models['M'+str(i)]

    # A helper function that returns degree of the tree with dictionary dic.
    def tree_deg(self, dic, done):
        return sum([done[w] * dic[w] for w in dic])

    # Helper function that multiplies trees. 
    # Given a dictionary {tree: power} outputs \prod_{tree} tree^power
    def trees_multiply(self, model, dic):
        
        trees = list(dic.keys())
        w1 = trees[0]
        
        if len(dic) == 1:  # If there is only one tree, it is faster to just return tree^n
            return model[w1] ** dic[w1]
        if len(dic) == 2:  # If only two unique trees is multiplied faster to return this
            w2 = trees[1]
            return (model[w1] ** dic[w1]) * (model[w2] ** dic[w2])

        tree_val = model[w1] ** dic[w1]
        for i in range(1,len(trees)):
            tree_val *= model[trees[i]]**dic[trees[i]]
        return tree_val
    
    # Creates all possible combinations of the values of the trees that can multiply planted trees.

    def extra_trees(self, W):
        trees_vals = self.R.values.copy()
        if 'xi' in self.R.degrees:
            trees_vals['xi'] = W
        dic_values = {}
        for i in self.R.rule_extra:
            dic_values[i] = self.trees_multiply(trees_vals, self.R.rule_extra[i])

        return dic_values

    # Given a realization of the noise W, creates a Model where all trees conform to the rule and are of degree <= 'deg'.
    def create_one_model(self, W, lollipop = None, extra_planted = None, extra_deg = None):

        # first let the model consist of the I[xi] only.
        if lollipop is None: # if lollipop is not given Integrate noise W
            model = self.I({'xi': W}, derivative = self.derivative)
        else: # otherwise simply add lollipop
            model ={'I[xi]' : lollipop}
        # 'planted' is a set that keeps track of the planted trees I[\tau].
        # 'done' is the dictionary that keeps track of all trees that were created together with their degree.

        planted, done = {'I[xi]'}, self.R.degrees.copy() #create set of planted trees and a dictinary of the trees degrees
        
        # Add planted trees that correspond to functions u^i for i in \mathcal{J}. 
        if extra_planted is not None: 
            model.update(extra_planted)
            planted = planted.union(set(extra_planted.keys()))
            done.update(extra_deg)
        # If necessary add spatial derivative of the I[xi] denoted by I'[xi]
        if self.derivative:
            planted.add("I'[xi]")
            done["I'[xi]"] = done["I[xi]"] - 1
            
        extra_trees_values = self.extra_trees(W)
        
        # Add trees of greater height
        for j in range(1, self.H):

            # Compute multiplications of the planted trees. self.R.max is the maximum possible width
            for k in range(1, self.R.max + 1):  # k is the number of trees multiplied
                # check all possible combinations of product of k planted trees
                for words in comb([w for w in planted], k):
                    tree, dic = self.R.words_to_tree(words)  # create one tree product out of the list of trees
                    temp_deg = self.tree_deg(dic, done) # compute the degree of this tree
                    # check if the tree needs to be added. k <= self.R.free_num checks if the product of k trees can exist
                    if tree not in done and tree not in self.R.exceptions and k <= self.R.free_num and temp_deg + self.R.degrees['I'] <= self.deg:
                        model[tree] = self.trees_multiply(model, dic)  # add this tree to the model
                        # if necessary add the tree multiplied by extra trees.
                    done[tree] = temp_deg  # include the tree to the done dictionary together with its degree
                    # multiply by the extra trees if such are present in the rule
                    for i in extra_trees_values: # add extra trees that correspond to multiplicative width
                        if k <= self.R.rule_power[i]: # check if extra tree can multiply the k product of planted trees
                            extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(i))
                            new_tree = extra_tree +'(' + tree +')' #shape of the new tree
                            deg = done[tree] + self.tree_deg(extra_dic, done) # degree of a new tree
                            if new_tree not in done and new_tree not in self.R.exceptions and deg <= self.deg:
                                if tree in model:
                                    model[new_tree] = model[tree]*extra_trees_values[i]
                                else:
                                    model[new_tree] = self.trees_multiply(model, dic)*extra_trees_values[i]
                                done[new_tree] = done[tree] + self.tree_deg(extra_dic, done)                
            
            # integrate trees from the previous iteration.
            this_round = self.I(model, planted, self.R.exceptions, self.derivative)  
            
            keys = [tree for tree in this_round.keys() if tree not in self.R.degrees and tree not in planted]
            
            # include theese integrated trees to the model. Don't include trees that are not of the form I[\tau]
            for IZ in keys:  
                if IZ[1] == "[":
                    Z = IZ[2:-1]  # IZ = I[Z]
                else:  
                    Z = IZ[3:-1]  # IZ = I'[Z]
                if Z not in planted and Z in model:
                    model.pop(Z)  # Delete Z tree from the model if it is not planted
                model[IZ] = this_round.pop(IZ)
                planted.add(IZ)  # add tree IZ to planted
                if IZ[1] == "[":
                    done[IZ] = done[Z] + self.R.degrees['I']  # add degree to IZ  
                else:  
                    done[IZ] = done[Z] + self.R.degrees['I'] - 1
        
        # delete all the trees of the form \partial_x I[\tau] and only keep
        # trees of the form I[\tau]
        
        if self.derivative:
            model = {IZ: model[IZ] for IZ in model if IZ[1] != "'"}

        return model

    # Create list of models given a df of noises W and a potential df of lolipops. It is slightly faster to 
    # create list of models where each model is a dictionary of dataframes rather that creating multilayer 
    # dataframe in a first place.
    def create_model_list(self, W, dt, diff = True, lollipops = None, extra_planted = None, extra_deg = None, key = None):

        num_noises = W.shape[0]
        
        if diff: 
            dW = np.zeros(W.shape)
            dW[:,1:,:] = np.diff(W, axis = 1)/dt
        else:
            dW = W*dt
            
        self.models = []
            
        for i in tqdm(range(num_noises)):
                
            if lollipops is None: # No precomputed values of I[dW] 
                if extra_planted is None: # No trees of the form u^i, i in \mathcal{J}
                    self.models.append(self.create_one_model(dW[i]))
                else: # Add u^i to the 0-th height of the model
                    self.models.append(self.create_one_model(dW[i], extra_planted = {key : extra_planted[i]}, extra_deg = {key : extra_deg}))
            else: # I[dW] is already given
                if extra_planted is None:
                    self.models.append(self.create_one_model(dW[i], lollipops[i]))
                else:
                    self.models.append(self.create_one_model(dW[i], lollipops[i], {key: extra_planted[i]}, {key : extra_deg}))
            
        self.size = num_noises
        
    # Create dataframes of models that only contain certain space time points and not the full model. 
    
    def create_model_points(self, W, points, dt, diff = True, lollipops = None, extra_planted = None, extra_deg = None, key = None):

        num_noises = W.shape[0]
        
        # differentiate noise/forcing to create dW 
        if diff:
            dW = np.zeros(W.shape)
            dW[:,1:,:] = np.diff(W, axis = 1)/dt
        else:
            dW = W*dt
        
        # dataframe {space-time points: models at this space time point for each realization of W}
        points_of_models = {}

        for i in tqdm(range(num_noises)):
            
            if lollipops is None: # No precomputed values of I[dW] 
                if extra_planted is None: # No trees of the form u^i, i in \mathcal{J}
                    M = self.create_one_model(dW[i])
                else: # Add u^i to the 0-th height of the model
                    if type(key) == str: # if only one key from \mathcal{J} and its not given as a dictionary {key: degree}
                        M = self.create_one_model(dW[i], extra_planted = {key : extra_planted[i]}, extra_deg = {key : extra_deg})
                    else: 
                        M = self.create_one_model(dW[i], extra_planted = {a : extra_planted[a][i] for a in key}, extra_deg = extra_deg)
            else: # I[dW] is already given
                if extra_planted is None:
                    M = self.create_one_model(dW[i], lollipops[i])
                else:
                    if type(key) == str:
                        M = self.create_one_model(dW[i], lollipops[i], {key: extra_planted[i]}, {key : extra_deg})
                    else:
                        M = self.create_one_model(dW[i], lollipops[i], {a : extra_planted[a][i] for a in key}, extra_deg)
            if i==0:
                # Initialize all possible trees in the model (extraxt model's feature set)
                trees = list(M.keys())
                # Initialize datafrae for each space-time point
                # Dataframes index is number of realization of the noises. Columns correspond to model feature set
                # and value of df[\tau].iloc[n] is value of f_\tau(t,x) for n-th realization of the noise and p = (t,x)
                points_of_models = {p: pd.DataFrame(index=np.arange(num_noises), columns = trees) for p in points}
                
            for p in points:
                points_of_models[p].iloc[i] = [M[t][p] for t in trees]
      
        self.size = num_noises
        
        return points_of_models

    def discrete_diff_2d(self, vec, N, axis, f = None, flatten = True, higher = True):
        a = vec.copy()
        if len(a.shape) == 1:
            a = a.reshape(len(vec)//N, N)
        if axis == 1:
            if f is None:
                if higher: # central approximation of a dervative
                    a[:,:-1,:] = (np.roll(a[:,:-1,:], -1, axis = 1) - np.roll(a[:,:-1,:], 1, axis = 1))/2
                else:
                    a[:,:-1,:] = a[:,:-1,:] - np.roll(a[:,:-1,:], 1, axis = 1)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:,:-1,:] = (self.vectorized(f, np.roll(a[:,:-1,:], -1, axis = 1)) - self.vectorized(f, np.roll(a[:,:-1,:], 1, axis = 1)))/2
                else:
                    a[:,:-1,:] = self.vectorized(f, a[:,:-1,:]) - self.vectorized(f, np.roll(a[:,:-1,:], 1, axis = 1))
            a[:,-1,:] = a[:,0,:] # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        if axis == 2:
            if f is None:
                if higher: # central approximation of a dervative
                    a[:,:,:-1] = (np.roll(a[:,:,:-1], -1, axis = 2) - np.roll(a[:,:,:-1], 1, axis = 2))/2
                else:
                    a[:,:,:-1] = a[:,:,:-1] - np.roll(a[:,:,:-1], 1, axis = 2)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:,:,:-1] = (self.vectorized(f, np.roll(a[:,:,:-1], -1, axis = 2)) - self.vectorized(f, np.roll(a[:,:,:-1], 1, axis = 2)))/2
                else:
                    a[:,:,:-1] = self.vectorized(f, a[:,:,:-1]) - self.vectorized(f, np.roll(a[:,:,:-1], 1, axis = 2))
            a[:,:,-1] = a[:,:,0] # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        return a

    def create_one_model_2d(self, W, X, lollipop = None, extra_planted = None, extra_deg = None):
        dx = X[1,0] - X[0,0]

        # first let the model consist of the I[xi] only.
        if lollipop is None: # if lollipop is not given Integrate noise W
            model = self.I({'xi': W}, derivative = self.derivative)
        else: # otherwise simply add lollipop
            model ={'I[xi]' : lollipop, 'I1[xi]': self.discrete_diff_2d(lollipop, X.shape[0], axis = 1, flatten = False, higher = False)/dx,\
                 'I2[xi]': self.discrete_diff_2d(lollipop, X.shape[0], axis = 2, flatten = False, higher = False)/dx}
        # 'planted' is a set that keeps track of the planted trees I[\tau].
        # 'done' is the dictionary that keeps track of all trees that were created together with their degree.

        planted, done = {'I[xi]'}, self.R.degrees.copy() #create set of planted trees and a dictinary of the trees degrees
        
        # Add planted trees that correspond to functions u^i for i in \mathcal{J}. 
        if extra_planted is not None: 
            model.update(extra_planted)
            planted = planted.union(set(extra_planted.keys()))
            done.update(extra_deg)
            if self.derivative:
                for arr in extra_planted.values():
                    model.update({'I1_c[u_0]': self.discrete_diff_2d(arr, X.shape[0], axis = 1, flatten = False, higher = False)/dx,\
                    'I2_c[u_0]': self.discrete_diff_2d(arr, X.shape[0], axis = 2, flatten = False, higher = False)/dx})
                    planted = planted.union({'I1_c[u_0]','I2_c[u_0]'})
                    done.update({'I1_c[u_0]':1,'I2_c[u_0]':1})
        # If necessary add spatial derivative of the I[xi] denoted by I'[xi]
        if self.derivative:
            planted.add("I1[xi]")
            planted.add("I2[xi]")
            done["I1[xi]"] = done["I[xi]"] - 1
            done["I2[xi]"] = done["I[xi]"] - 1
            
        extra_trees_values = self.extra_trees(W)
        
        # Add trees of greater height
        for j in range(1, self.H):

            # Compute multiplications of the planted trees. self.R.max is the maximum possible width
            for k in range(1, self.R.max + 1):  # k is the number of trees multiplied
                # check all possible combinations of product of k planted trees
                for words in comb([w for w in planted], k):
                    tree, dic = self.R.words_to_tree(words)  # create one tree product out of the list of trees
                    temp_deg = self.tree_deg(dic, done) # compute the degree of this tree
                    # check if the tree needs to be added. k <= self.R.free_num checks if the product of k trees can exist
                    if tree not in done and tree not in self.R.exceptions and k <= self.R.free_num and temp_deg + self.R.degrees['I'] <= self.deg:
                        model[tree] = self.trees_multiply(model, dic)  # add this tree to the model
                        # if necessary add the tree multiplied by extra trees.
                    done[tree] = temp_deg  # include the tree to the done dictionary together with its degree
                    # multiply by the extra trees if such are present in the rule
                    for i in extra_trees_values: # add extra trees that correspond to multiplicative width
                        if k <= self.R.rule_power[i]: # check if extra tree can multiply the k product of planted trees
                            extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(i))
                            new_tree = extra_tree +'(' + tree +')' #shape of the new tree
                            deg = done[tree] + self.tree_deg(extra_dic, done) # degree of a new tree
                            if new_tree not in done and new_tree not in self.R.exceptions and deg <= self.deg:
                                if tree in model:
                                    model[new_tree] = model[tree]*extra_trees_values[i]
                                else:
                                    model[new_tree] = self.trees_multiply(model, dic)*extra_trees_values[i]
                                done[new_tree] = done[tree] + self.tree_deg(extra_dic, done)                
            
            # integrate trees from the previous iteration.
            this_round = self.I(model, planted, self.R.exceptions, self.derivative)
            
            keys = [tree for tree in this_round.keys() if tree not in self.R.degrees and tree not in planted]
            
            # include theese integrated trees to the model. Don't include trees that are not of the form I[\tau]
            for IZ in keys:  
                if IZ[1] == "[":
                    Z = IZ[2:-1]  # IZ = I[Z]
                else:  
                    Z = IZ[3:-1]  # IZ = I'[Z]
                if Z not in planted and Z in model:
                    model.pop(Z)  # Delete Z tree from the model if it is not planted
                model[IZ] = this_round.pop(IZ)
                planted.add(IZ)  # add tree IZ to planted
                if IZ[1] == "[":
                    done[IZ] = done[Z] + self.R.degrees['I']  # add degree to IZ  
                else:  
                    done[IZ] = done[Z] + self.R.degrees['I'] - 1
        
        # delete all the trees of the form \partial_x I[\tau] and only keep
        # trees of the form I[\tau]
        
        if self.derivative:
            model = {IZ: model[IZ] for IZ in model if IZ[1] != "1" and IZ[1] != "2"}

        return model

    # Create dataframes of models that only contain certain space time points and not the full model. 
    
    def create_model_points_2d(self, W, points, X, dt, batch_size, diff = False, lollipops = None, extra_planted = None, extra_deg = None, key = None):

        num_noises = W.shape[0]
        
        # differentiate noise/forcing to create dW 
        if diff:
            dW = np.zeros(W.shape)
            dW[:,1:,:] = np.diff(W, axis = 1)/dt
        else:
            dW = W*dt
        
        # dataframe {space-time points: models at this space time point for each realization of W}
        points_of_models = {}

        for j in tqdm(range(int(num_noises/batch_size))):
            for i in range(batch_size):
                
                if lollipops is None: # No precomputed values of I[dW] 
                    if extra_planted is None: # No trees of the form u^i, i in \mathcal{J}
                        M = self.create_one_model_2d(dW[i+j*batch_size], X)
                    else: # Add u^i to the 0-th height of the model
                        if type(key) == str: # if only one key from \mathcal{J} and its not given as a dictionary {key: degree}
                            M = self.create_one_model_2d(dW[i+j*batch_size], X, extra_planted = {key : extra_planted[i+j*batch_size]}, extra_deg = {key : extra_deg})
                        else: 
                            M = self.create_one_model_2d(dW[i+j*batch_size], X, extra_planted = {a : extra_planted[a][i+j*batch_size] for a in key}, extra_deg = extra_deg)
                else: # I[dW] is already given
                    if extra_planted is None:
                        M = self.create_one_model_2d(dW[i+j*batch_size], X, lollipops[i+j*batch_size])
                    else:
                        if type(key) == str:
                            M = self.create_one_model_2d(dW[i+j*batch_size], X, lollipops[i+j*batch_size], {key: extra_planted[i+j*batch_size]}, {key : extra_deg})
                        else:
                            M = self.create_one_model_2d(dW[i+j*batch_size], X, lollipops[i+j*batch_size], {a : extra_planted[a][i+j*batch_size] for a in key}, extra_deg)
                if j==0 and i==0:
                    # Initialize all possible trees in the model (extraxt model's feature set)
                    trees = list(M.keys())
                    # Initialize datafrae for each space-time point
                    # Dataframes index is number of realization of the noises. Columns correspond to model feature set
                    # and value of df[\tau].iloc[n] is value of f_\tau(t,x) for n-th realization of the noise and p = (t,x)
                    points_of_models = {p: pd.DataFrame(index=np.arange(num_noises), columns = trees) for p in points}
                    
                for p in points:
                    points_of_models[p].iloc[i+j*batch_size] = [M[t][p] for t in trees]
          
        self.size = num_noises
        
        return points_of_models
        
    # If list of models is created convert it to the df of models
    def list_to_df(self):
        if type(self.models) is not list:
            print('Models are not of the list type')
            return
        trees = list(self.models[0].keys()) # trees in the model
        time, space = self.models[0][trees[0]].index, self.models[0][trees[0]].columns #
        models = pd.MultiIndex.from_product([['M'+str(i) for i in range(1,self.size+1)], trees, space])
        Models = pd.DataFrame(index=times, columns=models)
        model = pd.MultiIndex.from_product([keys, space])

        for i in tqdm(range(self.size)):
            M = pd.DataFrame(index=times, columns=model)
            for A in keys:
                M[A] = self.models[i][A]
            Models['M' + str(i + 1)] = M

        self.models = Models

    def save_models(self, name):

        if type(self.models) is list:
            print('Converting to DataFrame')
            self.list_to_df()
        elif self.models is None:
            print("Models are not yet created.")
            return
        else:
            self.models.to_csv(name)

    def upload_models(self, name):

        self.models = pd.read_csv(name, index_col=0, header=[0, 1, 2])
        trees = [m[0] for m in self.models['M1'].columns[::self.models.columns.levshape[2]]]  # extract trees
        self.models.columns = pd.MultiIndex.from_product([['M' + str(i + 1) for i in range(self.models.columns.levshape[0])], trees, np.asarray(self.models['M1'][trees[0]].columns, dtype=np.float16)])
