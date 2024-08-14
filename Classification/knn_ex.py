import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter



def distancia_euclideana(vet1, vet2):
  distancia = 0

  for i in range(len(vet1)-1):
    distancia += (vet1[i] - vet2[i])**2
  
  distancia = distancia**(1/2)
  return distancia

def knn_predict(X_train, X_test, y_train, y_test, k, p):
  y_hat_test = []

  for test_point in X_test:
      distances = []

      for train_point in X_train:
          distance = distancia_euclideana(test_point, train_point)
          distances.append(distance)
      
      # Store distances in a dataframe
      df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                              index=y_train.index)
      
      # Sort distances, and only consider the k closest points
      df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

      # Create counter object to track the labels of k closest neighbors
      counter = Counter(y_train[df_nn.index])

      # Get most common label of all the nearest neighbors
      prediction = counter.most_common()[0][0]
      
      # Append prediction to output list
      y_hat_test.append(prediction)
      
  return y_hat_test


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
  print('Knn')

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
  

  x = df.drop(target, axis=1)
  y = df[target]
  print("Total samples: {}".format(x.shape[0]))

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
  print("Total train samples: {}".format(X_train.shape[0]))
  print("Total test  samples: {}".format(X_test.shape[0]))

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  y_mush_test = knn_predict(X_train, X_test, y_train, y_test, k=4, p=2)

  accuracy = accuracy_score(y_test, y_mush_test)*100
  f1 = f1_score(y_test, y_mush_test, average='macro')
  print("Acurracy K-NN from scratch: {:.2f}%".format(accuracy))
  print("F1 Score K-NN from scratch: {:.2f}%".format(f1))

  cm = confusion_matrix(y_test, y_mush_test)
  plot_confusion_matrix(cm, y.unique(), False, "Confusion Matrix - K-NN")      
  plot_confusion_matrix(cm, y.unique(), True, "Confusion Matrix - K-NN normalized")
  plt.show()

if __name__ == '__main__':
  main()