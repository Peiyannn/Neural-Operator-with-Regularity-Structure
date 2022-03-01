# adapted from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures.git
# %%
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LassoLars, LassoLarsCV, Ridge, LassoCV, RidgeCV, HuberRegressor
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn import preprocessing
import seaborn as sns

err1 = lambda x, y: np.sqrt(mean_absolute_error(x,y))
err2 = lambda x, y: np.sqrt(mean_squared_error(x,y))

Error1 = lambda x, y: err1(x,y)/err1(np.zeros(y.shape), y)
Error2 = lambda x, y: err2(x,y)/err2(np.zeros(y.shape), y)



# %%
class summary():
    
    def __init__(self, test):
        
        self.test = test
    
    def time_comparison(self, k, t, Solution, Prediction, error = Error2, title = True):
        # Draw graphs of the k-th solution and the prediction at time t. X-axis: space grid. Y-axis: values.
        fig = plt.figure(figsize=(8, 6))
        if type(Solution) in {list, np.array}:
            T = Solution[0].index
            Solution[self.test[k]].iloc[t].plot(color = 'b', label = "Solution at time {}.".format(T[t]))
            S = Solution[self.test[k]].iloc[t].values
        else:
            T = Solution.index
            Solution["S"+str(self.test[k]+1)].iloc[t].plot(color = 'b', label = "Solution at time {}.".format(T[t]))
            S = Solution["S"+str(self.test[k]+1)].iloc[t].values
        if type(Prediction) in {list, np.array}:
            Prediction[k].iloc[t].plot(color = 'r', label = "Prediction at time {}.".format(T[t]))
            P = Prediction[k].iloc[t].values
        else:
            Prediction["S"+str(self.test[k]+1)].iloc[t].plot(color = 'r', label = "Prediction at time {}.".format(T[t]))
            P = Prediction["S"+str(self.test[k]+1)].iloc[t].values
        plt.xlabel('Space')
        plt.ylabel('Value')
        if title:
            plt.title("Solution and Prediction of the test case {}, at time point number {}.".format(k+1,t))
        plt.legend()
        plt.show()
        
        print("Error between solution and prediction at time {} is : {}.".format(T[t], error(P,S)))
        
    def space_comparison(self, k, x, Solution, Prediction, error = Error2, title = True):
        # Draw graphs of the k-th solution and the prediction at space point x. X-axis: time grid. Y-axis: values.
        fig = plt.figure(figsize=(8, 6))
        if type(Solution) in {list, np.array}:
            X = Solution[0].columns
            Solution[self.test[k]][X[x]].plot(color = 'b', label = "Solution at space point {}.".format(X[x]))
            S = Solution[self.test[k]][X[x]].values
        else:
            X = Solution["S1"].columns
            Solution["S"+str(self.test[k]+1)][X[x]].plot(color = 'b', label = "Solution at space point {}.".format(X[x]))
            S = Solution["S"+str(self.test[k]+1)][X[x]].values
        if type(Prediction) in {list, np.array}:
            Prediction[k][X[x]].plot(color = 'r', label = "Prediction at space point {}.".format(X[x]))
            P = Prediction[k][X[x]].values
        else:
            Prediction["S"+str(self.test[k]+1)][X[x]].plot(color = 'r', label = "Prediction at space point {}.".format(X[x]))
            P = Prediction["S"+str(self.test[k]+1)][X[x]].values
        plt.xlabel('Time')
        plt.ylabel('Value')
        if title:
            plt.title("Solution and Prediction of the test case {}, at space point number {}.".format(k+1,x))
        plt.legend()
        plt.show()
        
        print("Error between solution and prediction at space point {} is : {}.".format(X[x], error(P,S)))
        
    def full_comparison(self, k, Solution, Prediction, error = Error2, cmap = "coolwarm", show_title = False):
        
        # Heatmaps of k-th solution and prediction. All space-time values
        
        if type(Solution) in {list, np.array}:
            S = Solution[self.test[k]].values
        else:
            S = Solution["S"+str(self.test[k]+1)].values
        if type(Prediction) in {list, np.array}:
            P = Prediction[k].values
        else:
            P = Prediction["S"+str(self.test[k]+1)].values
        
        er = error(P, S)
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        if show_title:
            fig.suptitle('Heatmaps for solution and prediction. Test case {}. Relative l2 error: {}'.format(k+1 ,round(er,4)))
        sns.color_palette(cmap, as_cmap=True)
        sns.heatmap(np.array(S.T).astype(np.float64), ax=axs[0], 
                    xticklabels=20, yticklabels=40, cmap = cmap)
        axs[0].set_title("Solution", fontsize = 15)
        axs[0].set_xlabel('Time', fontsize = 15)
        axs[0].set_ylabel('Space', fontsize = 15)
        sns.heatmap(np.array(P.T).astype(np.float64), ax=axs[1], 
                    xticklabels=20, yticklabels=40, cmap = cmap)
        axs[1].set_title("Prediction", fontsize = 15)
        axs[1].set_xlabel('Time', fontsize = 15)
        axs[1].set_ylabel('Space', fontsize = 15)
        plt.show();
        if show_title is False:
            print('Error:', er)
        
    def errors_comparison(self, k, Solution, Prediction, error = Error2, cmap = "coolwarm", show_title = False):
        
        # Heatmaps for the errors of the k-th solution and prediction.
        
        if type(Solution) in {list, np.array}:
            S = Solution[self.test[k]].values
        else:
            S = Solution["S"+str(self.test[k]+1)].values
        if type(Prediction) in {list, np.array}:
            P = Prediction[k].values
        else:
            P = Prediction["S"+str(self.test[k]+1)].values

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        if show_title:
            fig.suptitle('Heatmaps for Absolute Errors and Relative l2 error of each point.')
        sns.color_palette("coolwarm", as_cmap=True)
        sns.heatmap(np.abs(np.array((S-P).T).astype(np.float64)), ax=axs[0], 
                    xticklabels=20, yticklabels=40, cmap = cmap);
        axs[0].set_title("Absolute Errors", fontsize = 15)
        axs[0].set_xlabel('Time', fontsize = 15)
        axs[0].set_ylabel('Space', fontsize = 15)
        std = (S-P)**2/S**2
        std = np.sqrt(np.array(std.T).astype(np.float64))
        
        sns.heatmap(std, ax=axs[1], xticklabels=20, yticklabels=40, cmap = cmap);
        axs[1].set_title("Relative l2 error at every point", fontsize = 15)
        axs[1].set_xlabel('Time', fontsize = 15)
        axs[1].set_ylabel('Space', fontsize = 15)
        plt.show()
        
    def one_error(self, k, Solution, Prediction, error = Error2):
        
        # Compute the error between the k-th solution and prediction
        
        if type(Solution) in {list, np.array}:
            S = Solution[self.test[k]].values
        else:
            S = Solution["S"+str(self.test[k]+1)].values
        if type(Prediction) in {list, np.array}:
            P = Prediction[k].values
        else:
            P = Prediction["S"+str(self.test[k]+1)].values
        return error(P,S)
    
    def raw_error(self, k, Solution, Prediction, raw = err2):
        
        # Compute the non scaled error between the k-th solution and prediction
        
        if type(Solution) in {list, np.array}:
            S = Solution[self.test[k]].values
        else:
            S = Solution["S"+str(self.test[k]+1)].values
        if Prediction == None:
            return raw(np.zeros(S.shape), S)
        elif type(Prediction) in {list, np.array}:
            P = Prediction[k].values
        else:
            P = Prediction["S"+str(self.test[k]+1)].values
        return raw(P,S)
    
    def errors(self, Solution, Prediction, raw = err2, full = True, show = True, maxi = False):
        
        # compute full errors between all solution and prediction
        
        er, l2, av_er, max_er = 0, 0, 0, 0

        for k in range(len(self.test)):
            er += self.raw_error(k, Solution, Prediction)
            l2 += self.raw_error(k, Solution, None)
            av_er += self.one_error(k, Solution, Prediction)
            max_er = max(max_er, self.one_error(k, Solution, Prediction))
        
        if full:
            if show: print("Total relative error for all test cases:", er/l2)
            return er/l2
        else:
            if show: print("Average error over all test cases:", av_er/len(self.test))
            if maxi: 
                print('')
                print("Maximum relative error among test cases:", max_er)
            return av_er/len(self.test)

# %%
