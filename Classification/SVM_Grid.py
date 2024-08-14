# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def main():

    input_file = 'DataSets/MushroomsNumeric.data'
    names = [ 
        'Poisonous',
        'Cap_Shape', 
        'Cap_Surface',
        'Cap_Color',
        'Bruises',
        'Odor',
        'Gill_Attachment',
        'Gill_Spacing',
        'Gill_Size',
        'Gill_Color',
        'Stalk_Shape',
        'Stalk_Surface_Above_Ring',
        'Stalk_Surface_Bellow_Ring',
        'Stalk_Color_Above_Ring',
        'Stalk_Color_Bellow_Ring',
        'Veil_Type',
        'Veil_Color',
        'Ring_Number',
        'Ring_Type',
        'Spore_Print_Color',
        'Population',
        'Habitat'
        ]
    
    target = 'Poisonous'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                        names = names) # Nome das colunas    

    # Separate X and y data
    X = df.drop(target, axis=1)
    y = df[target]
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)


    # predict using test dataset
    y_hat_test = clf.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}%".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, df[target].unique(), False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, df[target].unique(), True, "Confusion Matrix - SVM sklearn normalized" )  
    plt.show()


if __name__ == "__main__":
    main()