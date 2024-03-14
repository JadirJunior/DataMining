import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def main():
    # Faz a leitura do arquivo
    input_file = 'Datasets/MushroomsNumeric.data'
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
    ShowInformationDataFrame(df,"Dataframe original")

    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z_score normalization
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