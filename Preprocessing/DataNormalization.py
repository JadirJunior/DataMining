import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
    y = df.loc[:,[target]].values

    # Z-score normalization
    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    # Mix-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

    
if __name__ == "__main__":
    main()