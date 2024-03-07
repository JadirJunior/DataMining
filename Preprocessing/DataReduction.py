import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = 'Datasets/Mushrooms.data'
    names = [ 
            'Poisonous',
            'Cap-Shape', 
            'Cap-Surface',
            'Cap-Color',
            'Bruises',
            'Odor',
            'Gill-Attachment',
            'Gill-Spacing',
            'Gill-Size',
            'Gill-Color',
            'Stalk-Shape',
            'Stalk-Surface-Above-Ring',
            'Stalk-Surface-Bellow-Ring',
            'Stalk-Color-Above-Ring',
            'Stalk-Color-Bellow-Ring',
            'Veil-Type',
            'Veil-Color',
            'Ring-Number',
            'Ring-Type',
            'Spore-Print-Color',
            'Population',
            'Habitat'
    ]
    features = [
            'Cap-Shape', 
            'Cap-Surface',
            'Cap-Color',
            'Bruises',
            'Odor',
            'Gill-Attachment',
            'Gill-Spacing',
            'Gill-Size',
            'Gill-Color',
            'Stalk-Shape',
            'Stalk-Surface-Above-Ring',
            'Stalk-Surface-Bellow-Ring',
            'Stalk-Color-Above-Ring',
            'Stalk-Color-Bellow-Ring',
            'Veil-Type',
            'Veil-Color',
            'Ring-Number',
            'Ring-Type',
            'Spore-Print-Color',
            'Population',
            'Habitat'
    ]

    target = 'Poisonous'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")


    
    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:, [target]].values


    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalizedDf,"Dataframe Normalized")

    # PCA projection
    pca = PCA()    
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(data = principalComponents[:,0:2], 
                               columns = ['principal component 1', 
                                          'principal component 2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)    
    ShowInformationDataFrame(finalDf,"Dataframe PCA")
    
    VisualizePcaProjection(finalDf, target)
    

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")
    
           
def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['e', 'p', ]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()