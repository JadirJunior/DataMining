from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    features = [
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
   
    # Separating out the features
    X = df.loc[:, features].values
    print(X.shape)

    # Separating out the target
    y = df.loc[:,[target]].values

    # Standardizing the features
    X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
    print(X_train.shape)
    print(X_test.shape)

    clf = DecisionTreeClassifier(max_leaf_nodes=13)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()
    
    predictions = clf.predict(X_test)
    print(predictions)
    
    result = clf.score(X_test, y_test)
    print('Acuraccy:')
    print(result)


if __name__ == "__main__":
    main()