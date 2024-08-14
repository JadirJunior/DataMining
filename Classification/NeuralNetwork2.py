import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# def plot_training_history(history):
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Erro treino')
#     plt.plot(history.history['val_loss'], label='Erro teste')
#     plt.title('Histórico de Treinamento')
#     plt.ylabel('Função de custo')
#     plt.xlabel('Época de treinamento')
#     plt.legend()
#     plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico da perda (erro)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Erro treino')
    plt.plot(history.history['val_loss'], label='Erro teste')
    plt.title('Histórico de Perda')
    plt.xlabel('Época de treinamento')
    plt.ylabel('Função de custo')
    plt.legend()
    
    # Gráfico da acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia teste')
    plt.title('Histórico de Acurácia')
    plt.xlabel('Época de treinamento')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


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
        'Stalk_Surface_Below_Ring',
        'Stalk_Color_Above_Ring',
        'Stalk_Color_Below_Ring',
        'Veil_Type',
        'Veil_Color',
        'Ring_Number',
        'Ring_Type',
        'Spore_Print_Color',
        'Population',
        'Habitat'
        ]
    
    target = 'Poisonous'
    df = pd.read_csv(input_file, names=names)
    
    # Converte a coluna target em valores numéricos (0 ou 1)
    df[target] = df[target].map({'e': 0, 'p': 1})  # 'e' para comestível, 'p' para venenoso

    # Seleciona os dados de entrada e saída
    X = df.drop(target, axis=1)
    y = df[target]

    print("Total samples: {}".format(X.shape[0]))

    # Divide em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cria e treina o modelo de Rede Neural
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=21))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    # Plot do histórico de treinamento
    plot_training_history(history)

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Erro no conjunto de teste: {:.2f}'.format(loss))
    print('Acurácia no conjunto de teste: {:.2f}%'.format(accuracy * 100))

if __name__ == "__main__":
    main()