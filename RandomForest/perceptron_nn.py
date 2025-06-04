import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_layers=None, learning_rate=0.01, activation_function='linear'):
        """
        Inicjalizacja sieci neuronowej dla problemu regresji
        
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
            Funkcja aktywacji ('sigmoid', 'relu', 'tanh')
        """
        # Słownik funkcji aktywacji
        self.activation_functions = {
            'sigmoid': (self._sigmoid, self._sigmoid_derivative),
            'relu': (self._relu, self._relu_derivative),
            'tanh': (self._tanh, self._tanh_derivative),
            'linear': (self._linear, self._linear_derivative)
        }
        
        # Sprawdzenie poprawności zadanej funkcji aktywacji
        if activation_function not in self.activation_functions:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation_function}. "
                             f"Dostępne opcje: {list(self.activation_functions.keys())}")
        
        # Wybór funkcji aktywacji
        self.activation_fn, self.activation_derivative = self.activation_functions[activation_function]
        
        # Parametry uczenia
        self.learning_rate = learning_rate
        
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
            while current > 1:
                current = max(current - 1, 1)
                layers.append(current)
        elif isinstance(num_layers, int):
            # Liczba warstw podana jako liczba całkowita
            layers = []
            current = num_inputs
            for _ in range(num_layers):
                current = max(current - 1, 1)
                layers.append(current)
        else:
            # Bezpośrednia lista liczby neuronów w warstwach
            layers = list(num_layers)
        
        # Wyświetlenie informacji o architekturze sieci
        print(f"Architektura sieci: Wejścia({num_inputs}) -> ", end="")
        for i, layer in enumerate(layers):
            print(f"Warstwa_{i+1}({layer}) -> ", end="")
        print("Wyjście(1)")
        
        # Inicjalizacja wag dla wielowarstwowej sieci
        self.weights = []
        prev_layer_size = num_inputs
        
        for layer_size in layers:
            # Xavier/Glorot initialization for better numerical stability
            limit = np.sqrt(6 / (prev_layer_size + layer_size))
            layer_weights = np.random.uniform(-limit, limit, (layer_size, prev_layer_size + 1))
            self.weights.append(layer_weights)
            prev_layer_size = layer_size
        
        # Dodanie warstwy wyjściowej (jeden neuron dla regresji)
        limit = np.sqrt(6 / (prev_layer_size + 1))
        output_weights = np.random.uniform(-limit, limit, (1, prev_layer_size + 1))
        self.weights.append(output_weights)
    
    # Funkcje aktywacji i ich pochodne
    def _linear(self, x):
        return x
    
    def _linear_derivative(self, x):
        return np.ones_like(x)

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
    
    def _add_bias(self, X):
        """Dodanie kolumny bias do macierzy wejściowej"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def forward_propagation(self, X):
        """
        Propagacja w przód przez wielowarstwową sieć
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane wejściowe
        
        Returns:
        --------
        tuple: (lista wyjść warstw, finalne wyjście)
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
        
        # Warstwa wyjściowa (liniowa)
        final_input_with_bias = self._add_bias(current_layer_input)
        final_output = np.dot(final_input_with_bias, self.weights[-1].T)
        
        return layer_outputs, final_output
    
    def backpropagation(self, X, y, layer_outputs, final_output):
        """
        Propagacja wsteczna i aktualizacja wag
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane wejściowe
        y : numpy.ndarray
            Wartości docelowe
        layer_outputs : list
            Wyjścia z warstw ukrytych
        final_output : numpy.ndarray
            Finalne przewidywania
        """
        # Obliczenie błędu
        output_error = final_output - y
        
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
    
    def fit(self, X, y, X_val=None, y_val=None, num_epochs=1000, verbose=False, 
            early_stopping=True, patience=50, min_delta=0.0001, convergence_threshold=1e-6):
        """
        Trening sieci neuronowej
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane treningowe
        y : numpy.ndarray
            Wartości docelowe
        X_val : numpy.ndarray, optional
            Dane walidacyjne wejściowe (wymagane dla early stopping)
        y_val : numpy.ndarray, optional
            Dane walidacyjne docelowe (wymagane dla early stopping)
        num_epochs : int, optional
            Liczba epok treningu
        verbose : bool, optional
            Wyświetlanie informacji o postępie treningu
        early_stopping : bool, optional
            Czy używać early stopping (domyślnie True)
        patience : int, optional
            Liczba epok bez poprawy, po której trening zostanie zatrzymany (domyślnie 10)
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
        y = y.astype(np.float64)
        
        # Zapewnienie odpowiedniego kształtu y
        y = y.reshape(-1, 1)
        
        # Przygotowanie dla early stopping
        best_loss = float('inf')
        best_weights = None
        counter = 0
        
        # Weryfikacja danych walidacyjnych
        if early_stopping:
                # Ensure validation data is properly formatted
                X_val = X_val.astype(np.float64)
                y_val = y_val.astype(np.float64)
                y_val = y_val.reshape(-1, 1)
        
        # Historia treningu
        history = {}
        
        # Trening
        for epoch in range(num_epochs):
            history["epoch"] = epoch+1
            # Propagacja w przód
            layer_outputs, final_output = self.forward_propagation(X)
            self.history = history
            
            # Obliczenie błędu na zbiorze treningowym
            train_loss = np.mean((final_output - y)**2)
            train_mae = np.mean(abs((final_output - y)/y))
            
            history['train_mse'] = train_loss
            history['train_mae'] = train_mae

            # Check for NaN and report error if found
            if np.isnan(train_loss):
                additional_info = f"Warning: NaN loss detected at epoch {epoch + 1}. Training stopped."
                print(additional_info)
                # Restore best weights if available
                if not best_weights:
                    self.weights = best_weights
                
                self.history = history
                return history
                
            # Check if loss is extremely small (convergence achieved)
            if train_loss < convergence_threshold:
                additional_info = f"Convergence achieved at epoch {epoch + 1}. Training MSE: {train_loss:.8f}"
                print(additional_info)

                _, val_predictions = self.forward_propagation(X_val)
                val_loss = np.mean((val_predictions - y_val)**2)
                history['val_mse'] =val_loss

                self.history = history
                return history
            
            # Propagacja wsteczna
            self.backpropagation(X, y, layer_outputs, final_output)
            
            # Early stopping check
            if early_stopping:
                # Obliczenie błędu walidacji
                _, val_predictions = self.forward_propagation(X_val)
                val_loss = np.mean((val_predictions - y_val)**2)
                history['val_mse'] =val_loss
                
                # Sprawdzenie czy jest poprawa
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]  # Zapisanie najlepszych wag
                    counter = 0  # Reset licznika
                else:
                    counter += 1
                
                # Sprawdzenie cierpliwości
                if counter >= patience:
                    additional_info = f'Early stopping na epoce {epoch + 1}/{num_epochs}. {patience} epok bez poprawy. Najlepszy val_MSE: {best_loss:.4f}'
                    print(additional_info)
                    # Przywrócenie najlepszych wag
                    self.weights = best_weights
                    self.history = history
                    return history
                
                # Wyświetlanie informacji o postępie
                if verbose and (epoch + 1) % 100 == 0:
                    print(f'Epoka {epoch + 1}/{num_epochs}, Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}')
            else:
                # Wyświetlanie informacji o postępie bez walidacji
                if verbose and (epoch + 1) % 100 == 0:
                    print(f'Epoka {epoch + 1}/{num_epochs}, Train MSE: {train_loss:.4f}')
        
        # Przywrócenie najlepszych wag, jeśli były używane
        if early_stopping and best_weights:
            self.weights = best_weights
            
        self.history = history

        return history
    
    def predict(self, X):
        """
        Przewidywanie wartości dla nowych danych
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dane do przewidywania
        
        Returns:
        --------
        numpy.ndarray: Przewidywane wartości
        """
        # Ensure inputs are float64 for consistency
        X = X.astype(np.float64)
        _, final_output = self.forward_propagation(X)
        return final_output
    






############ do zastanowienia się
def cross_validate_mlp(X, y, model_params, num_folds=5, num_epochs=2000, 
                       early_stopping=True, patience=25, min_delta=0.0001, verbose=False):
    """
    Wykonuje k-krotną walidację krzyżową dla MultiLayerPerceptron
    
    Parameters:
    -----------
    X : numpy.ndarray
        Dane wejściowe
    y : numpy.ndarray
        Wartości docelowe
    model_params : dict
        Parametry do inicjalizacji MultiLayerPerceptron
    num_folds : int
        Liczba foldów w k-krotnej walidacji
    num_epochs : int
        Liczba epok treningu
    early_stopping : bool
        Czy używać early stopping
    patience : int
        Liczba epok bez poprawy dla early stopping
    min_delta : float
        Minimalna poprawa MSE
    verbose : bool
        Czy wypisywać postęp
    
    Returns:
    --------
    dict:
        Średnie i poszczególne wyniki MSE i MAE z walidacji
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    val_mse_list = []
    val_mae_list = []
    
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Inicjalizacja nowego modelu dla każdego folda
        model = MultiLayerPerceptron(num_inputs=X.shape[1], **model_params)
        
        if verbose:
            print(f"\n=== Fold {fold}/{num_folds} ===")
        
        model.fit(X_train, y_train,
                  X_val=X_val, y_val=y_val,
                  num_epochs=num_epochs,
                  early_stopping=early_stopping,
                  patience=patience,
                  min_delta=min_delta,
                  verbose=verbose)
        
        # Przewidywanie na danych walidacyjnych
        y_pred = model.predict(X_val)
        
        # Metryki
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        val_mse_list.append(mse)
        val_mae_list.append(mae)
        
        if verbose:
            print(f"Fold {fold} MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        fold += 1
    
    return {
        'val_mse_mean': np.mean(val_mse_list),
        'val_mae_mean': np.mean(val_mae_list),
        'val_mse_std': np.std(val_mse_list),
        'val_mae_std': np.std(val_mae_list),
        'val_mse_all': val_mse_list,
        'val_mae_all': val_mae_list
    }
