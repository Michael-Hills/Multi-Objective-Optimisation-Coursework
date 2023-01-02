import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.tree
import sklearn.ensemble
import pymoo.util.nds.non_dominated_sorting as nds
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
from hyperopt import fmin,space_eval,partial,Trials,tpe,STATUS_OK,hp
import time


def TPR(y_true,y_pred):
    """Calcuates true positive rate"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tp/(tp+fn)

def FPR(y_true,y_pred):
    """Calculates false positive rate"""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   
    return fp / (fp + tn)



def data_preprocessing(test_size, number_features):
    """Performs the data preposessing"""

    #loads the breast cancer dataset
    data = sklearn.datasets.load_breast_cancer(as_frame=True)
    features = data.feature_names.tolist()

    df = data.frame

    #create new column for the classification
    df['Classification'] = data['target'].replace(
        {1: 'benign', 0: 'malignant'}
    )

    #Normalise the data
    scale = StandardScaler()
    df[features] = scale.fit_transform(df[features])


    #Random forest to get most important features and decrease model complexity
    clf = sklearn.ensemble.RandomForestClassifier(random_state=100)
    clf.fit(df[features], df['Classification'])
    feature_importances = pd.Series(
        list(clf.feature_importances_),
        index=features
    ).sort_values(ascending=False)


    importantFeatures = feature_importances[0:number_features].index.tolist()


    #Create test train splits
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            df[importantFeatures],
            df['Classification'],
            test_size=test_size,
            random_state=100,
            stratify=df['Classification']

        )
    
    
    return X_train, X_test, y_train, y_test


def singleObjective(X,y):
    """Implementation of the single objective classifier"""
    
    #Parameters to optimise
    parameter_space = {
    'hidden_layer_sizes': [(5,25,5),(5,50,5),(5,100,5),(10,25,10),(10,50,10),(10,100,10),(15,25,15),(15,50,15),(15,100,15)],
    'activation': ['tanh', 'relu','identity'],
    'alpha': [0.0001,0.5],
    'batch_size' : [200,250]
    }
    
    #Create the classifier model and do a grid search
    mlp = MLPClassifier(max_iter=10000,random_state=100)
    gs = GridSearchCV(mlp,parameter_space, scoring = 'accuracy',n_jobs=-1, cv=5)
    gs.fit(X, y)

    return gs.best_params_,gs


def multiObjective(X_train, y_train):
    """Implementation of the classifier with multiple performance metric objectives"""

    #parameters to optimise
    parameter_space = {
    'hidden_layer_sizes': [(5,25,5),(5,50,5),(5,100,5),(10,25,10),(10,50,10),(10,100,10),(15,25,15),(15,50,15),(15,100,15)],
    'activation': ['tanh', 'relu','identity'],
    'alpha': [0.0001,0.5],
    'batch_size' : [200,250]
    }


    #the performance metrics (the objectives)
    scoring = {
        'Accuracy': 'accuracy',
        'True Positive Rate': sklearn.metrics.make_scorer(TPR),
        'False Positive Rate': sklearn.metrics.make_scorer(FPR),
    }

    #create the classifier and grid search
    mlp = MLPClassifier(max_iter=10000,random_state=100)
    gs = GridSearchCV(mlp,parameter_space, scoring = scoring ,n_jobs=-1, cv=5,refit=False)
    gs.fit(X_train, y_train)

   
    #add results to a dataframe
    df = pd.DataFrame(gs.cv_results_['params'])
    df['Accuracy'] = gs.cv_results_['mean_test_Accuracy']
    df['True Positive Rate'] = gs.cv_results_['mean_test_True Positive Rate']
    df['False Positive Rate'] = gs.cv_results_['mean_test_False Positive Rate']
    

    return df


def plotObjectives(df,nonDom):
    """Plot the solutions showing the non-dominated solutions"""

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    
    x_vals = df['Accuracy'].values
    y_vals = df['True Positive Rate'].values
    z_vals = df['False Positive Rate'].values

    count = 0

    #make non dominated red and the rest blue
    for i in range(len(df)):
        if count in nonDom:
            ax.scatter(x_vals[i], y_vals[i], z_vals[i],color="red")

        else: 
            ax.scatter(x_vals[i], y_vals[i], z_vals[i],color="blue")

        count += 1


    ax.set_xlabel("Accuracy")
    ax.set_ylabel("TPR")
    ax.set_zlabel("FPR")
    plt.show()

    return


def findNonDominated(df):
    """Find the non-dominated solutions"""

    #the the metrics/objectives
    objs = ['Accuracy',
    'True Positive Rate',
    'False Positive Rate',]
    df_sorting = df.copy()


    #Invert the sign of the metrics where maximum is best
    df_sorting['Accuracy']= -1.0 * df_sorting['Accuracy']
    df_sorting['True Positive Rate'] = -1.0 * df_sorting['True Positive Rate']


    #Find the non dominated solutions
    nondom_idx = nds.find_non_dominated(df_sorting[objs].values)

    nonDom = df.iloc[nondom_idx].copy()

    plotObjectives(df_sorting,nondom_idx)

    return nonDom
    

def findBest(df):
    """Find the best solution by combining into a single objective"""

    acc = df['Accuracy'].values
    tpr = df['True Positive Rate'].values
    fpr = df['False Positive Rate'].values

    max = 0
    index = 0
    for i in range (len(acc)):
        val = 0.33* acc[i] + 0.33*tpr[i] + 0.33*-1*fpr[i]
        if val >= max:
            max = val
            index = i


    return df.iloc[index,:]



def testResultPredictions(params,X_train,X_test,y_train,y_test):
    """Function to test the prediction performance"""


    #create the model with the found best parameters
    clf = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        batch_size=params['batch_size'],
        alpha = params['alpha'],
        max_iter=10000,
        random_state=100
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)


    print('Test Accuracy:', sklearn.metrics.accuracy_score(
        y_test, pred)
    )

    print("Test TPR: ", TPR(y_test,pred))
    print("Test FPR: ", FPR(y_test,pred))

    print(" ")

    conf = confusion_matrix(y_test,pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
    cm_display.plot()
    plt.show()


    return

def objective(params,X_train, X_test, y_train, y_test):
    """
    Objective function to minimise
    """

    clf = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        batch_size=params['batch_size'],
        alpha = params['alpha'],
        max_iter=10000,
        random_state=100
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)


    acc =  sklearn.metrics.accuracy_score(y_test, pred)

    tpr =  TPR(y_test,pred)
    fpr =  FPR(y_test,pred)
    
    loss = 0.33*-1* acc+ 0.33*-1*tpr + 0.33*fpr
    
    return {'loss': loss,  'status': STATUS_OK}
    


def bayesianSearch(X_train, X_test, y_train, y_test, space, maxEvals):
    """
    Function to perform Bayesian search by minimising the objective function
    """
    
    trials = Trials()
    fmin_objective = partial(objective,X_train= X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=maxEvals, 
                trials=trials)

    best = space_eval(space, best)
    print ('best:')
    print (best)

    bestclf = MLPClassifier(
        hidden_layer_sizes=best['hidden_layer_sizes'],
        activation=best['activation'],
        batch_size=best['batch_size'],
        alpha = best['alpha'],
        max_iter=10000,
        random_state=100
    )
    

    
    return best, bestclf, trials

def getBayesianScores(X_train, X_test, y_train, y_test):
    """
    Function that runs the bayesian search
    """  

    pd.set_option('display.max_rows', None)
    hspace = {
        "hidden_layer_sizes": hp.choice('hidden_layer_sizes',(range(5,50,5),range(10,300,10),range(5,50,5))),
        "activation":hp.choice('activation', ['tanh', 'relu','identity']),
        "alpha": hp.choice('alpha',np.arange(0.0001,0.1,0.001)),
        "batch_size": hp.choice('batch_size',range(200,300,20))}

    best_params, bestclf, trials_use = bayesianSearch(X_train, X_test, y_train, y_test,space=hspace,maxEvals=60)

    return best_params,bestclf




X_train, X_test, y_train, y_test = data_preprocessing(0.2,5)

print("1. Single Objective")
print("2. Multi Objective Pareto Front")
print("3. Multi Objective Bayesian Optimisation")

while True:
    option = input("Enter option: ")

    if option == '1':

        start_time = time.perf_counter()

        singleParams, gs = singleObjective(X_train, y_train)


        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)


        print("--------Single Objective--------")
        print(" ")
        print("Best Parameters for Single Objective: ",gs.best_params_)
        print('Single Objective Train Accuracy:', gs.best_score_)

        testResultPredictions(singleParams,X_train, X_test, y_train, y_test)

        

        break
    elif option == '2':

        start_time = time.perf_counter()


        print("--------Multiple Objectives--------")
        print(" ")

        df_all = multiObjective(X_train, y_train)
        print("Total No. of Solutions: ",len(df_all.index))

        nonDom_df = findNonDominated(df_all)


        print("No. of non dominated solutions:", len(nonDom_df.index))

        print(nonDom_df)
        print(" ")


        multiParams = findBest(nonDom_df)
        print(multiParams)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)

        testResultPredictions(multiParams,X_train, X_test, y_train, y_test)

        break

    elif option == '3':

        start_time = time.perf_counter()
        bestParams, bestclf = getBayesianScores(X_train, X_test, y_train, y_test)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)

        testResultPredictions(bestParams,X_train, X_test, y_train, y_test)

        break

    else:
        print("Please enter a number between 1 and 3")
