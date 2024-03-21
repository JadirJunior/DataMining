#Data handling Imports
import pandas as pd 
import numpy as np

#Visualizing Data Imports
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
  
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



    input_file = 'DataSets/MushroomsNumeric.data'



    df = pd.read_csv(input_file,
                        names = names)
    

    num_col = names
    columns = 3
    f, axes = plt.subplots(8, columns, figsize=(20,5), sharey = True) 
    
    column = 0
    cont = 1
    line = 0

    for i,col in enumerate(num_col):
        print(col)
        s = sb.countplot(x=col, data = df, hue='Poisonous', alpha=0.7, ax=axes[line][column])
        s.legend(loc="upper right", prop={'size': 4})

        res = cont % 8
        print(res)
        if ( res == 0):
            column += 1
            line = 0
        else:
            line += 1
        cont += 1
        
        
        for p in s.patches:
            s.annotate(format(p.get_height(), '.0f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'center', 
            xytext = (0, 5), 
            textcoords = 'offset points')

    plt.show()

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

if __name__ == "__main__":
    main()