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
  

  
  input_file = 'DataSets/Mushrooms.data'

  df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas    


  plt.rcParams['figure.figsize']=15,5
  plt.subplot(121)
  plt.title('Mushroom Class Type Count', fontsize=10)
  s = sb.countplot(x = "Poisonous", data = df, alpha=0.7)
  for p in s.patches:
      s.annotate(format(p.get_height(), '.1f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha = 'center', va = 'center', 
                  xytext = (0, 4), 
                  textcoords = 'offset points')
      
  ax = plt.subplot(122)
  mush_classpie = df['Poisonous'].value_counts()
  mush_size = mush_classpie.values.tolist()
  mush_types = mush_classpie.axes[0].tolist()
  mush_labels = 'Edible', 'Poisonous'
  colors = ['#EAFFD0', '#F38181']
  plt.title('Mushroom Class Type Percentange', fontsize=10)
  patches, texts, autotexts = plt.pie(mush_size, labels=mush_labels, colors=colors,
          autopct='%1.1f%%', shadow=True, startangle=150)
  for text,autotext in zip(texts,autotexts):
      text.set_fontsize(14)
      autotext.set_fontsize(14)

  plt.axis('equal')  
  plt.show()


if __name__ == "__main__":
    main()