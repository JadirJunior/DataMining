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
    output_file = 'DataSets/Mushrooms.data'
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
    # for c in df.columns:
    #     if c != 'Poisonous':
    #         convert_categoric_numeric(df, c)
    allClasses = setClasses()

    # df['Cap-Shape'] = df['Cap-Shape'].map(allClasses[1])

    for class_map, column_name in zip(allClasses, df.columns):
        df[column_name] = df[column_name].map(class_map)

    

    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)  
    

def setClasses():
    Poisonous = dict(p = 'Poisonous', e = 'Edible')
    Cap_Shape = dict(b='bell', c='conical', x='convex', f='flat', k='knobbed', s='sunken')
    Cap_Surface = dict(f='fibrous', g='grooves', y='scaly', s='smooth')
    Cap_Color = dict(n='brown', b='buff', c='cinnamon', g='gray', r='green', p='pink', u='purple', e='red', w='white', y='yellow')
    Gill_Attachment = dict(a='attached', d='descending', f='free', n='notched')
    Gill_Spacing = dict(c='close', w='crowded', d='distant')
    Gill_Size = dict(b='broad', n='narrow')
    Gill_Color = dict(n='brown', b='buff', g='gray', r='green', p='pink', u='purple', o='orange',h='chocolate', k='black', e='red', w='white', y='yellow')
    Stalk_Shape = dict(e='enlarging',t='tapering')
    Stalk_Surface_Above_Ring = dict(f='fibrous',y='scaly',k='silky',s='smooth')
    Stalk_Surface_Bellow_Ring = dict(f='fibrous',y='scaly',k='silky',s='smooth')
    Stalk_Color_Above_Ring = dict(n='brown', b='buff', c='cinnamon', g='gray', o='orange', p='pink', e='red',w='white', y='yellow')
    Stalk_Color_Bellow_Ring = dict(n='brown', b='buff', c='cinnamon', g='gray', o='orange', p='pink', e='red',w='white', y='yellow')
    Veil_Color = dict(n='brown', o='orange', w='white', y='yellow')
    Ring_Number = dict(n='None', o='One', t='Two')
    Ring_Type = dict(c='cobwebby', e='evanescent', f='flaring', l='large', n='none', p='pendant', s='sheathing', z='zone')
    Bruises = dict(t='True', f='False')
    Odor = dict(a='almond', l='anise', c='creosote', y='fishy', f='foul', m='musty', n='none', p='pungent', s='spicy')
    Spore_Print_Color = dict(k='black', n='brown', b='buff', h='chocoloate', r='green', o='Orange', u='Purple',w='White', y='Yellow')
    Population = dict(a='abundant', c='clustered', n='numerous', s='scattered', v='several', y='solitary')
    Habitat = dict(g='grasses', l='leaves', m='meadows', p='paths', u='urban', w='waste', d='woods')
    Veil_Type = dict(p = 'partial',u = 'universal')


    allClasses = [
            Poisonous, 
            Cap_Shape, 
            Cap_Surface,
            Cap_Color,
            Bruises,
            Odor,
            Gill_Attachment,
            Gill_Spacing,
            Gill_Size,
            Gill_Color,
            Stalk_Shape,
            Stalk_Surface_Above_Ring,
            Stalk_Surface_Bellow_Ring,
            Stalk_Color_Above_Ring,
            Stalk_Color_Bellow_Ring,
            Veil_Type,
            Veil_Color,
            Ring_Number,
            Ring_Type,
            Spore_Print_Color,
            Population,
            Habitat
        ]
    
    return allClasses
    

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