import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
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
            'Stalk_Root',
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
            #'Stalk_Root',
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
    output_file = 'DataSets/MushroomsNumeric.data'
    input_file = 'DataSets/agaricus-lepiota.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
    
    df_original = df.copy()
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    pd.set_option("display.max_columns", None) #Mostra todas as colunas requeridas.
    print(df.head(150))
    print("\n")        

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")    
    
    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)
    method = 'mode' # number or median or mean or mode
    
    for c in columns_missing_value:
        UpdateMissingValues(df, c)
    
    
    # Code for change categorical values into numeric
    for c in df.columns:
        if c != 'Poisonous':
            convert_categoric_numeric(df, c)
    

    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)  
    

def UpdateMissingValues(df, column, method="mode", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)


def convert_categoric_numeric(df, column):
    df[column] = pd.factorize(df[column])[0]


if __name__ == "__main__":
    main()