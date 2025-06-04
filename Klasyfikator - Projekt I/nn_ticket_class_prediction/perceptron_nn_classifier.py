import numpy as np

class BinaryNeuralNetworkClassifier:
    def __init__(self, num_inputs, num_layers=None, learning_rate=0.01, activation_function='relu'):
        """
        Inicjalizacja sieci neuronowej dla problemu klasyfikacji binarnej
        
        Parameters:
        -----------
        num_inputs : int
            Liczba zmiennych wejściowych
        num_layers : int or list, optional
            - Jeśli None: automatyczne tworzenie warstw zmniejszających się liczbę neuronów
            - Jeśli int: liczba warstw ukrytych
            - Jeśli lista: dokładna liczba neuronów w każdej warstwie ukrytej
        learning_rate : float, optional
            Współczynnik uczenia się (domyślnie 0.01)
        activation_function : str, optional
            Funkcja aktywacji ('sigmoid', 'relu', 'tanh', 'leaky_relu', 'elu', 'swish')
        """
        # Słownik funkcji aktywacji
        self.activation_functions = {
            'sigmoid': (self._sigmoid, self._sigmoid_derivative),
            'relu': (self._relu, self._relu_derivative),
            'tanh': (self._tanh, self._tanh_derivative),
            'leaky_relu': (self._leaky_relu, self._leaky_relu_derivative),
            'elu': (self._elu, self._elu_derivative),
            'swish': (self._swish, self._swish_derivative)
        }
        
        # Sprawdzenie poprawności zadanej funkcji aktywacji
        if activation_function not in self.activation_functions:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation_function}. "
                             f"Dostępne opcje: {list(self.activation_functions.keys())}")
        
        # Wybór funkcji aktywacji
        self.activation_fn, self.activation_derivative = self.activation_functions[activation_function]
        
        # Parametry uczenia
        self.learning_rate = learning_rate
        self.num_classes = 2  # Binary classification - always 2 classes
        
        # Konfiguracja warstw
        self._configure_layers(num_inputs, num_layers)

        self.history = None
    
    def _configure_layers(self, num_inputs, num_layers):
        """
        Konfiguracja architektury sieci neuronowej
        
        Parameters:
        -----------
        num_inputs : int
            Liczba zmiennych wejściowych
        num_layers : int or list or None
            Specyfikacja warstw sieci
        """
        # Domyślna strategia tworzenia warstw, jeśli nie podano
        if num_layers is None:
            # Automatyczne tworzenie warstw malejących
            layers = []
            current = num_inputs
            while current > 4 and len(layers) < 3:  # Stop at 4 neurons minimum
                current = max(current // 2, 4)
                if current > 1:  # Ensure we don't add layers smaller than output
                    layers.append(current)
        elif isinstance(num_layers, int):
            # Liczba warstw podana jako liczba całkowita
            layers = []
            current = num_inputs
            for _ in range(num_layers):
                current = max(current // 2, 4)
                if current > 1:
                    layers.append(current)
        else:
            # Bezpośrednia lista liczby neuronów w warstwach
            layers = list(num_layers)
        
        # Wyświetlenie informacji o architekturze sieci
        print(f"Architektura sieci binarnej: Wejścia({num_inputs}) -> ", end="")
        for i, layer in enumerate(layers):
            print(f"Warstwa_{i+1}({layer}) -> ", end="")
        print(f"Wyjście(1 neuron - klasyfikacja binarna)")
        
        # Inicjalizacja wag dla wielowarstwowej sieci
        self.weights = []
        prev_layer_size = num_inputs
        
        for layer_size in layers:
            # Xavier/Glorot initialization for better numerical stability
            limit = np.sqrt(6 / (prev_layer_size + layer_size))
            layer_weights = np.random.uniform(-limit, limit, (layer_size, prev_layer_size + 1))
            self.weights.append(layer_weights)
            prev_layer_size = layer_size
        
        # Dodanie warstwy wyjściowej (1 neuron dla klasyfikacji binarnej)
        limit = np.sqrt(6 / (prev_layer_size + 1))
        output_weights = np.random.uniform(-limit, limit, (1, prev_layer_size + 1))
        self.weights.append(output_weights)
    
    # Funkcje aktywacji i ich pochodne
    def _sigmoid(self, x):
        """Funkcja aktywacji sigmoid z zabezpieczeniem przed przepełnieniem"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))
    
    def _sigmoid_derivative(self, x):
        """Pochodna funkcji sigmoid"""
        return x * (1.0 - x)
    
    def _relu(self, x):
        """Funkcja aktywacji ReLU"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Pochodna funkcji ReLU"""
        return np.where(x > 0, 1.0, 0.0)
    
    def _tanh(self, x):
        """Funkcja aktywacji tanh"""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Pochodna funkcji tanh"""
        return 1.0 - np.power(x, 2)
    
    def _leaky_relu(self, x, alpha=0.01):
        """Funkcja aktywacji Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)
    
    def _leaky_relu_derivative(self, x, alpha=0.01):
        """Pochodna funkcji Leaky ReLU"""
        return np.where(x > 0, 1.0, alpha)
    
    def _elu(self, x, alpha=1.0):
        """Funkcja aktywacji ELU (Exponential Linear Unit)"""
        return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -250, 250)) - 1))
    
    def _elu_derivative(self, x, alpha=1.0):
        """Pochodna funkcji ELU"""
        return np.where(x > 0, 1.0, alpha * np.exp(np.clip(x, -250, 250)))
    
    def _swish(self, x):
        """Funkcja aktywacji Swish (x * sigmoid(x))"""
        sigmoid_x = self._sigmoid(x)
        return x * sigmoid_x
    
    def _swish_derivative(self, x):
        """Pochodna funkcji Swish"""
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    
    def _add_bias(self, X):
        """Dodanie kolumny bias do macierzy wejściowej"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def forward_propagation(self, X):
        """
        Propagacja w przód przez wielowarstwową sieć (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane wejściowe
        
        Returns:
        --------
        tuple: (lista wyjść warstw, prawdopodobieństwo klasy pozytywnej)
        """
        # Lista do przechowywania wyjść z każdej warstwy
        layer_outputs = []
        current_layer_input = X
        
        # Propagacja przez warstwy ukryte
        for layer_weights in self.weights[:-1]:
            # Dodanie bias do danych wejściowych
            current_layer_input_with_bias = self._add_bias(current_layer_input)
            
            # Obliczenie wejścia do warstwy
            layer_input = np.dot(current_layer_input_with_bias, layer_weights.T)
            
            # Aktywacja 
            layer_output = self.activation_fn(layer_input)
            
            # Dodaj do listy wyjść i przygotuj dla następnej warstwy
            layer_outputs.append(layer_output)
            current_layer_input = layer_output
        
        # Warstwa wyjściowa (sigmoid dla klasyfikacji binarnej)
        final_input_with_bias = self._add_bias(current_layer_input)
        logits = np.dot(final_input_with_bias, self.weights[-1].T)
        final_output = self._sigmoid(logits)  # Sigmoid dla klasyfikacji binarnej
        
        return layer_outputs, final_output
    
    def backpropagation(self, X, y_binary, layer_outputs, final_output):
        """
        Propagacja wsteczna i aktualizacja wag (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane wejściowe
        y_binary : numpy.ndarray
            Wartości docelowe binarne (0 lub 1)
        layer_outputs : list
            Wyjścia z warstw ukrytych
        final_output : numpy.ndarray
            Finalne przewidywania (prawdopodobieństwa)
        """
        # Obliczenie błędu dla sigmoid i binary cross-entropy
        output_error = final_output - y_binary.reshape(-1, 1)
        
        # Przygotowanie listy wejść dla każdej warstwy (z dodanym bias)
        layer_inputs_with_bias = []
        
        # Pierwsza warstwa ma jako wejście X
        layer_inputs_with_bias.append(self._add_bias(X))
        
        # Dla każdej kolejnej warstwy, wejściem jest wyjście poprzedniej warstwy
        for i in range(len(layer_outputs)-1):
            layer_inputs_with_bias.append(self._add_bias(layer_outputs[i]))
        
        # Dla warstwy wyjściowej wejściem jest wyjście ostatniej warstwy ukrytej
        if layer_outputs:
            layer_inputs_with_bias.append(self._add_bias(layer_outputs[-1]))
        
        # Propagacja wsteczna błędu
        current_error = output_error
        
        # Iteracja przez warstwy od końca
        for i in range(len(self.weights)-1, -1, -1):
            # Pobranie wejścia dla danej warstwy
            layer_input_with_bias = layer_inputs_with_bias[i]
            
            # Obliczenie gradientu dla wag
            gradient = np.dot(current_error.T, layer_input_with_bias)
            
            # Aktualizacja wag z klipowaniem gradientu dla stabilności
            clipped_gradient = np.clip(gradient, -1.0, 1.0)  # Prevent exploding gradients
            self.weights[i] = self.weights[i] - self.learning_rate * clipped_gradient
            
            # Obliczenie błędu dla poprzedniej warstwy (jeśli nie jest to pierwsza warstwa)
            if i > 0:
                # Błąd propagowany do warstwy poprzedniej (bez bias)
                error_without_bias = np.dot(current_error, self.weights[i][:, 1:])
                
                # Zastosowanie pochodnej funkcji aktywacji
                if i > 1:  # Dla warstw ukrytych
                    error_without_bias = error_without_bias * self.activation_derivative(layer_outputs[i-1])
                else:  # Dla pierwszej warstwy (która ma jako wejście X)
                    if len(layer_outputs) > 0:
                        error_without_bias = error_without_bias * self.activation_derivative(layer_outputs[0])
                
                current_error = error_without_bias
    
    def _binary_cross_entropy_loss(self, y_true, y_pred):
        """Obliczenie straty binary cross-entropy"""
        # Zabezpieczenie przed log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _accuracy(self, y_true, y_pred):
        """Obliczenie dokładności klasyfikacji binarnej"""
        y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(y_true == y_pred_binary)
    
    def fit(self, X, y, X_val=None, y_val=None, num_epochs=1000, verbose=False, 
            early_stopping=True, patience=50, min_delta=0.0001, convergence_threshold=1e-6):
        """
        Trening sieci neuronowej (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane treningowe
        y : numpy.ndarray
            Etykiety klas binarne (0 lub 1)
        X_val : numpy.ndarray, optional
            Dane walidacyjne wejściowe (wymagane dla early stopping)
        y_val : numpy.ndarray, optional
            Etykiety walidacyjne binarne (0 lub 1) (wymagane dla early stopping)
        num_epochs : int, optional
            Liczba epok treningu
        verbose : bool, optional
            Wyświetlanie informacji o postępie treningu
        early_stopping : bool, optional
            Czy używać early stopping (domyślnie True)
        patience : int, optional
            Liczba epok bez poprawy, po której trening zostanie zatrzymany (domyślnie 50)
        min_delta : float, optional
            Minimalna poprawa uznawana za znaczącą (domyślnie 0.0001)
        convergence_threshold : float, optional
            Próg straty, poniżej którego uznajemy, że model osiągnął zbieżność (domyślnie 1e-6)
        
        Returns:
        --------
        dict: Historia treningu z metrykami
        """
        # Ensure inputs are float64 for better precision
        X = X.astype(np.float64)
        y = y.astype(np.float64)  # y should be binary (0 or 1)
        
        # Przygotowanie dla early stopping
        best_loss = float('inf')
        best_weights = None
        counter = 0
        
        # Weryfikacja danych walidacyjnych
        if early_stopping:
            X_val = X_val.astype(np.float64)
            y_val = y_val.astype(np.float64)  # y_val should be binary (0 or 1)
        
        # Historia treningu
        history = {}
        
        # Trening
        for epoch in range(num_epochs):
            history["epoch"] = epoch+1
            # Propagacja w przód
            layer_outputs, final_output = self.forward_propagation(X)
            self.history = history
            
            # Obliczenie błędu na zbiorze treningowym
            train_loss = self._binary_cross_entropy_loss(y, final_output.flatten())
            train_accuracy = self._accuracy(y, final_output)
            
            history['train_loss'] = train_loss
            history['train_accuracy'] = train_accuracy

            # Check for NaN and report error if found
            if np.isnan(train_loss):
                additional_info = f"Warning: NaN loss detected at epoch {epoch + 1}. Training stopped."
                print(additional_info)
                # Restore best weights if available
                if best_weights:
                    self.weights = best_weights
                
                self.history = history
                return history
                
            # Check if loss is extremely small (convergence achieved)
            if train_loss < convergence_threshold:
                additional_info = f"Convergence achieved at epoch {epoch + 1}. Training Loss: {train_loss:.8f}"
                print(additional_info)

                if early_stopping:
                    _, val_predictions = self.forward_propagation(X_val)
                    val_loss = self._binary_cross_entropy_loss(y_val, val_predictions.flatten())
                    val_accuracy = self._accuracy(y_val, val_predictions)
                    history['val_loss'] = val_loss
                    history['val_accuracy'] = val_accuracy

                self.history = history
                return history
            
            # Propagacja wsteczna
            self.backpropagation(X, y, layer_outputs, final_output)
            
            # Early stopping check
            if early_stopping:
                # Obliczenie błędu walidacji
                _, val_predictions = self.forward_propagation(X_val)
                val_loss = self._binary_cross_entropy_loss(y_val, val_predictions.flatten())
                val_accuracy = self._accuracy(y_val, val_predictions)
                history['val_loss'] = val_loss
                history['val_accuracy'] = val_accuracy
                
                # Sprawdzenie czy jest poprawa
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]  # Zapisanie najlepszych wag
                    counter = 0  # Reset licznika
                else:
                    counter += 1
                
                # Sprawdzenie cierpliwości
                if counter >= patience:
                    additional_info = f'Early stopping na epoce {epoch + 1}/{num_epochs}. {patience} epok bez poprawy. Najlepszy val_loss: {best_loss:.4f}'
                    print(additional_info)
                    # Przywrócenie najlepszych wag
                    self.weights = best_weights
                    self.history = history
                    return history
                
                # Wyświetlanie informacji o postępie
                if verbose and (epoch + 1) % 100 == 0:
                    print(f'Epoka {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            else:
                # Wyświetlanie informacji o postępie bez walidacji
                if verbose and (epoch + 1) % 100 == 0:
                    print(f'Epoka {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        
        # Przywrócenie najlepszych wag, jeśli były używane
        if early_stopping and best_weights:
            self.weights = best_weights
            
        self.history = history

        return history
    
    def predict(self, X):
        """
        Przewidywanie klas dla nowych danych (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane do przewidywania
        
        Returns:
        --------
        numpy.ndarray: Przewidywane klasy (0 lub 1)
        """
        # Ensure inputs are float64 for consistency
        X = X.astype(np.float64)
        _, probabilities = self.forward_propagation(X)
        return (probabilities >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Przewidywanie prawdopodobieństw klas dla nowych danych (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane do przewidywania
        
        Returns:
        --------
        numpy.ndarray: Prawdopodobieństwa dla klasy pozytywnej (shape: (n_samples,))
        """
        # Ensure inputs are float64 for consistency
        X = X.astype(np.float64)
        _, probabilities = self.forward_propagation(X)
        return probabilities.flatten()
    
    def predict_proba_both_classes(self, X):
        """
        Przewidywanie prawdopodobieństw dla obu klas (klasyfikacja binarna)
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane do przewidywania
        
        Returns:
        --------
        numpy.ndarray: Prawdopodobieństwa [P(class=0), P(class=1)] dla każdej próbki (shape: (n_samples, 2))
        """
        # Ensure inputs are float64 for consistency
        X = X.astype(np.float64)
        _, probabilities = self.forward_propagation(X)
        prob_positive = probabilities.flatten()
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])