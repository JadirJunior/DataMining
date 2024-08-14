import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.random.randn(21)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.best_weights = None
        self.best_bias = None
        self.lowest_error = float('inf')

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                
                cumulative_errors.append(cumulative_error)
                
                # Check if this is the best model so far
                if cumulative_error < self.lowest_error:
                    self.lowest_error = cumulative_error
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias

        return cumulative_errors

    def evaluate_accuracy(self, input_vectors, targets):
        correct_predictions = 0
        for input_vector, target in zip(input_vectors, targets):
            prediction = self.predict(input_vector)
            predicted_class = 1 if prediction >= 0.5 else 0
            if predicted_class == target:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(targets)
        return accuracy


def main():
    learning_rate = 0.1
    neural_network = NeuralNetwork(learning_rate)

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

    x = df.drop(target, axis=1)
    targets = df[target]

    # Normaliza os dados
    x = StandardScaler().fit_transform(x)

    # Treina a rede neural
    training_error = neural_network.train(x, targets.values, 10000)

    # Plota o erro de treinamento
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.show()

    # Avalia a acurácia com o melhor modelo
    neural_network.weights = neural_network.best_weights
    neural_network.bias = neural_network.best_bias
    accuracy = neural_network.evaluate_accuracy(x, targets.values)
    
    print(f"Acurácia da rede neural: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
