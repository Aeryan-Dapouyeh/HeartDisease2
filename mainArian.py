# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:51:41 2021

@author: Arian
"""

### This should be the main python code

# %% Import packages/libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import model_selection, tree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.stats import kstest, norm, zscore
import statsmodels.api as sm
import pylab
from toolbox_02450 import feature_selector_lr, bmplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch


from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, hist, legend, ylim
import sklearn.linear_model as lm

# %% Import data

# Read
#my_data = np.genfromtxt('heart.csv', names=header, delimiter=',')
df=pd.read_csv('heart.csv')

# Split
data=df.loc[:,'age':'thal']


N, M = df.shape

X = df.to_numpy()
y=(df.iloc[:,-1]).to_numpy()
y.shape = (y.shape[0], 1)

attributeNames = list(data.columns)


# %% Inspired by 8.1.2


# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma



# %%

# Fit regularized logistic regression model to training data to predict 
# the type of wine
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]


# %%


plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 4])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    





# %%  Inspired by 8.2.2
plt.rcParams.update({'font.size': 12})


classNames = ["0", "1"]

# K-fold CrossValidation (4 folds here to speed up this example)
K = 10
CV = model_selection.KFold(K,shuffle=True)

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units = 4 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 3000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
errors = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y[test_index] )
    
    # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) # store error rate for current CV fold 
    
    
    # Make a subplot for current cross validation fold that displays the 
    # decision boundary over the original data, "background color" corresponds
    # to the output of the sigmoidal transfer function (i.e. before threshold),
    # white areas are areas of uncertainty, and a deaper red/blue means 
    # that the network "is more sure" of a given class.
    
    
    
    ### THE ERROR IS RIGHT DOWN HERE!
    '''
    plt.figure(decision_boundaries.number)
    plt.subplot(subplot_size_1,subplot_size_2,k+1)
    plt.title('CV fold {0}'.format(k+1),color=color_list[k])
    predict = lambda x: net(torch.tensor(x, dtype=torch.float)).data.numpy()
    visualize_decision_boundary(predict, X, y, # provide data, along with function for prediction
                                attributeNames, classNames, # provide information on attribute and class names
                                train=train_index, test=test_index, # provide information on partioning
                                show_legend=k==(K-1)) # only display legend for last plot
    
    '''
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')
    
# Show the plots
# plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))



### Keep training models until you get one very good one.
### It's because of the randomness of the weights you initialize.



# %% Inspired by 6.2.1

### The following line seems to fix the weird error
X = zscore(X, ddof=1)

K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))





# %% 



k=0
for train_index, test_index in CV.split(X):

    #print(train_index)
    #print(test_index)
    
    
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 5
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    #print(Error_train_nofeatures[k])
    #print(Error_test_nofeatures[k])
    
    ### THE ERROR IS HERE!
    m = lm.LogisticRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    
    print(Error_train[k])
    print(Error_test[k])
    
    
    k+=1
    
    

# %% 
















# %% 

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 5
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LogisticRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    # textout = ''
    # selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    # Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    # if len(selected_features) == 0:
    #     print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    # else:
    #    m = lm.lm.LogisticRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
    #    Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
    #    Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
    #    figure(k)
    #    subplot(1,2,1)
    #    plot(range(1,len(loss_record)), loss_record[1:])
    #    xlabel('Iteration')
    #    ylabel('Squared error (crossvalidation)')    
        
    #    subplot(1,3,3)
    #    bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
    #    clim(-1.5,0)
    #    xlabel('Iteration')

    #print('Cross validation fold {0}/{1}'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}'.format(test_index))
    # print('Features no: {0}\n'.format(selected_features.size))

    k+=1


print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))





















# %% Linear regression

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y


# %% 


# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

# %% Logistic regression


model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 


misclass_rate = np.sum(y_est != y) / float(len(y_est))

print(misclass_rate)


# %% Plot the logistic regression model like ex.5_2_6(Doesn't work yet)


f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object (wine sample)'); ylabel('Predicted prob. of class White');
legend(['White', 'Red'])
ylim(-0.01,1.5)

show()


# %% 


test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)


model = lm.LogisticRegression()
model = model.fit(X_train,y_train)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X_test)
y_est_white_prob = model.predict_proba(X_test)[:, 0] 


misclass_rate = np.sum(y_est != y_test) / float(len(y_est))

print(misclass_rate)


