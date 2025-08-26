#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import json
import os
import random
import math
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import logging

from preprocess import load_adult_ac1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class NeuFairConfig:
    """Configuration class for NeuFair neuron dropping"""
    
    def __init__(self, model_architecture: Dict = None, neuron_states: Dict = None):
        self.model_architecture = model_architecture or {}
        self.neuron_states = neuron_states or {}
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_data = {
            'model_architecture': self.model_architecture,
            'neuron_states': self.neuron_states
        }
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        return cls(
            model_architecture=config_data['model_architecture'],
            neuron_states=config_data['neuron_states']
        )
    
    def get_layer_mask(self, layer_name: str) -> Optional[np.ndarray]:
        """Get mask for a specific layer"""
        if layer_name in self.neuron_states:
            return np.array(self.neuron_states[layer_name])
        return None

class SimulatedAnnealingRepair:
    """TensorFlow/Keras implementation of NeuFair using Simulated Annealing"""
    
    def __init__(self, model_path: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                 protected_attribs: List[int] = None, max_iter: int = 1000,
                 temp_schedule: str = 'linear', fairness_weight: float = 0.5):
        
        self.model_path = model_path
        self.original_model = tf.keras.models.load_model(model_path)
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        self.constraint = constraint
        self.protected_attribs = protected_attribs if protected_attribs else [8]
        self.max_iter = max_iter
        self.temp_schedule = temp_schedule
        self.fairness_weight = fairness_weight
        
        # Extract dense layers for neuron dropping
        self.dense_layers = self._get_dense_layers()
        self.layer_shapes = self._get_layer_shapes()
        
        # Initialize state
        self.current_state = self._initialize_state()
        self.best_state = self.current_state.copy()
        self.best_cost = float('inf')
        
        logger.info(f"Initialized SA repair with {len(self.dense_layers)} dense layers")
        for i, layer in enumerate(self.dense_layers):
            logger.info(f"Layer {i} ({layer.name}): {self.layer_shapes[i]} neurons")
    
    def _get_dense_layers(self) -> List[tf.keras.layers.Dense]:
        """Extract all dense layers from the model"""
        dense_layers = []
        for layer in self.original_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Skip the final output layer
                if layer != self.original_model.layers[-1]:
                    dense_layers.append(layer)
        return dense_layers
    
    def _get_layer_shapes(self) -> List[int]:
        """Get the number of neurons in each dense layer"""
        return [layer.units for layer in self.dense_layers]
    
    def _initialize_state(self) -> Dict[str, List[int]]:
        """Initialize the neuron dropping state (all neurons active)"""
        state = {}
        for i, layer in enumerate(self.dense_layers):
            # 1 means active, 0 means dropped
            state[layer.name] = [1] * layer.units
        return state
    
    def _create_masked_model(self, state: Dict[str, List[int]]) -> tf.keras.Model:
        """Create a new model with neurons masked according to state"""
        try:
            # Use clone_model instead of from_config for better compatibility
            masked_model = tf.keras.models.clone_model(self.original_model)
            masked_model.set_weights(self.original_model.get_weights())
            
            # Apply neuron masking to dense layers
            layer_name_to_index = {layer.name: idx for idx, layer in enumerate(masked_model.layers)}
            
            for layer_name, neuron_states in state.items():
                if layer_name in layer_name_to_index:
                    layer_idx = layer_name_to_index[layer_name]
                    layer = masked_model.layers[layer_idx]
                    
                    if isinstance(layer, tf.keras.layers.Dense):
                        mask = np.array(neuron_states, dtype=np.float32)
                        weights = layer.get_weights()
                        
                        if len(weights) >= 2:  # weights and biases
                            # Mask biases for this layer
                            weights[1] = weights[1] * mask
                            layer.set_weights(weights)
                            
                            # Mask input weights of the next dense layer
                            for next_idx in range(layer_idx + 1, len(masked_model.layers)):
                                next_layer = masked_model.layers[next_idx]
                                if isinstance(next_layer, tf.keras.layers.Dense):
                                    next_weights = next_layer.get_weights()
                                    if len(next_weights) >= 1:
                                        # Mask input weights (rows correspond to previous layer outputs)
                                        next_weights[0] = next_weights[0] * mask[:, np.newaxis]
                                        next_layer.set_weights(next_weights)
                                    break  # Only mask the immediate next dense layer
            
            return masked_model
            
        except Exception as e:
            logger.error(f"Error creating masked model: {e}")
            raise
    
    def compute_fairness(self, state: Dict[str, List[int]]) -> Tuple[float, float, float]:
        """Compute fairness metrics for a given neuron dropping state"""
        try:
            # Create temporary masked model
            masked_model = self._create_masked_model(state)
            
            # Evaluate on test set
            y_pred_proba = masked_model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = self.y_test.flatten()
            
            # Compute accuracy and F1 score
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Compute individual discrimination rate
            fairness_score = self._compute_individual_discrimination(masked_model)
            
            # Cleanup
            del masked_model
            tf.keras.backend.clear_session()
            
            return accuracy, f1, fairness_score
            
        except Exception as e:
            logger.warning(f"Error in compute_fairness: {e}")
            return 0.0, 0.0, 1.0  # Poor performance if computation fails
    
    def _compute_individual_discrimination(self, model, num_samples: int = 500) -> float:
        """Compute individual discrimination rate using random sampling"""
        discriminatory_count = 0
        total_samples = 0
        
        for _ in range(num_samples):
            # Generate random instance
            x = np.array([
                np.random.randint(self.constraint[i][0], self.constraint[i][1] + 1) 
                for i in range(len(self.constraint))
            ]).astype(np.float32)
            
            # Create similar instances by varying protected attributes
            similar_instances = self._generate_similar_instances(x)
            
            if self._is_discriminatory(x, similar_instances, model):
                discriminatory_count += 1
            total_samples += 1
        
        return discriminatory_count / total_samples if total_samples > 0 else 0.0
    
    def _generate_similar_instances(self, x: np.ndarray) -> np.ndarray:
        """Generate similar instances by varying protected attributes"""
        similar_instances = []
        
        # Generate all combinations of protected attribute values
        protected_domains = []
        for attr_idx in self.protected_attribs:
            min_val, max_val = self.constraint[attr_idx]
            protected_domains.append(list(range(min_val, max_val + 1)))
        
        import itertools
        for combination in itertools.product(*protected_domains):
            x_new = x.copy()
            for attr_idx, value in zip(self.protected_attribs, combination):
                x_new[attr_idx] = value
            similar_instances.append(x_new)
        
        return np.array(similar_instances)
    
    def _is_discriminatory(self, x: np.ndarray, similar_instances: np.ndarray, model) -> bool:
        """Check if an instance leads to discriminatory predictions"""
        x_input = x.reshape(1, -1)
        pred_original = (model.predict(x_input, verbose=0) > 0.5).astype(int)[0][0]
        
        for x_similar in similar_instances:
            x_similar_input = x_similar.reshape(1, -1)
            pred_similar = (model.predict(x_similar_input, verbose=0) > 0.5).astype(int)[0][0]
            
            if pred_similar != pred_original:
                return True
        
        return False
    
    def _cost_function(self, accuracy: float, f1: float, fairness_score: float) -> float:
        """Compute the cost function balancing accuracy and fairness"""
        # Lower fairness_score is better (less discrimination)
        # Higher accuracy and f1 are better
        performance_score = (accuracy + f1) / 2
        cost = self.fairness_weight * fairness_score + (1 - self.fairness_weight) * (1 - performance_score)
        return cost
    
    def _generate_neighbor_state(self, current_state: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Generate a neighboring state by flipping a random neuron"""
        new_state = {k: v.copy() for k, v in current_state.items()}
        
        # Randomly select a layer and neuron
        layer_names = list(new_state.keys())
        selected_layer = random.choice(layer_names)
        selected_neuron = random.randint(0, len(new_state[selected_layer]) - 1)
        
        # Flip the neuron state (with bias towards keeping neurons active)
        if new_state[selected_layer][selected_neuron] == 1:
            # Drop neuron with probability 0.3
            if random.random() < 0.3:
                new_state[selected_layer][selected_neuron] = 0
        else:
            # Activate neuron with probability 0.7
            if random.random() < 0.7:
                new_state[selected_layer][selected_neuron] = 1
        
        return new_state
    
    def _calculate_temperature(self, iteration: int) -> float:
        """Calculate temperature for simulated annealing"""
        initial_temp = 1.0
        final_temp = 0.01
        
        if self.temp_schedule == 'linear':
            return initial_temp - (initial_temp - final_temp) * (iteration / self.max_iter)
        elif self.temp_schedule == 'exponential':
            return initial_temp * (final_temp / initial_temp) ** (iteration / self.max_iter)
        elif self.temp_schedule == 'logarithmic':
            return initial_temp / (1 + math.log(1 + iteration))
        else:
            return initial_temp * (0.95 ** iteration)  # Default exponential decay
    
    def run_sa(self) -> NeuFairConfig:
        """Run simulated annealing optimization"""
        logger.info("Starting simulated annealing optimization...")
        
        # Evaluate initial state
        accuracy, f1, fairness = self.compute_fairness(self.current_state)
        self.best_cost = self._cost_function(accuracy, f1, fairness)
        current_cost = self.best_cost
        
        logger.info(f"Initial: Accuracy={accuracy:.4f}, F1={f1:.4f}, Fairness={fairness:.4f}, Cost={self.best_cost:.4f}")
        
        for iteration in range(self.max_iter):
            # Generate neighbor state
            neighbor_state = self._generate_neighbor_state(self.current_state)
            
            # Evaluate neighbor
            neighbor_accuracy, neighbor_f1, neighbor_fairness = self.compute_fairness(neighbor_state)
            neighbor_cost = self._cost_function(neighbor_accuracy, neighbor_f1, neighbor_fairness)
            
            # Calculate acceptance probability
            temperature = self._calculate_temperature(iteration)
            delta_cost = neighbor_cost - current_cost
            
            if delta_cost < 0 or (temperature > 0 and random.random() < math.exp(-delta_cost / temperature)):
                # Accept the neighbor
                self.current_state = neighbor_state
                current_cost = neighbor_cost
                
                # Update best state if necessary
                if neighbor_cost < self.best_cost:
                    self.best_state = {k: v.copy() for k, v in neighbor_state.items()}
                    self.best_cost = neighbor_cost
                    logger.info(f"New best at iteration {iteration}: "
                              f"Accuracy={neighbor_accuracy:.4f}, F1={neighbor_f1:.4f}, "
                              f"Fairness={neighbor_fairness:.4f}, Cost={self.best_cost:.4f}")
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iter}: "
                          f"Temp={temperature:.4f}, Current Cost={current_cost:.4f}, "
                          f"Best Cost={self.best_cost:.4f}")
        
        # Create and return configuration
        model_architecture = {
            'layer_names': [layer.name for layer in self.dense_layers],
            'layer_shapes': self.layer_shapes,
            'total_layers': len(self.dense_layers)
        }
        
        config = NeuFairConfig(
            model_architecture=model_architecture,
            neuron_states=self.best_state
        )
        
        logger.info("Simulated annealing completed!")
        return config

def create_fairer_model(original_model_path: str, config_path: str, output_path: str) -> str:
    """
    Create a fairer model by permanently applying neuron dropping configuration
    
    Args:
        original_model_path: Path to the original model
        config_path: Path to the NeuFair configuration file
        output_path: Path where the new fairer model should be saved
        
    Returns:
        Path to the created fairer model
    """
    logger.info("Creating fairer model from configuration...")
    
    # Load original model and configuration
    original_model = tf.keras.models.load_model(original_model_path)
    config = NeuFairConfig.load(config_path)
    
    # Use clone_model for better compatibility
    try:
        fairer_model = tf.keras.models.clone_model(original_model)
        fairer_model.set_weights(original_model.get_weights())
    except Exception as e:
        logger.error(f"Error cloning model: {e}")
        # Fallback: try to rebuild the model manually
        fairer_model = tf.keras.models.Sequential()
        for layer in original_model.layers:
            if hasattr(layer, 'get_config'):
                try:
                    layer_config = layer.get_config()
                    layer_class = type(layer)
                    new_layer = layer_class.from_config(layer_config)
                    fairer_model.add(new_layer)
                except Exception as layer_error:
                    logger.warning(f"Could not clone layer {layer.name}: {layer_error}")
                    fairer_model.add(layer)
        
        fairer_model.set_weights(original_model.get_weights())
    
    # Apply permanent neuron masking
    layer_name_to_index = {layer.name: idx for idx, layer in enumerate(fairer_model.layers)}
    
    for layer_name, neuron_states in config.neuron_states.items():
        if layer_name in layer_name_to_index:
            layer_idx = layer_name_to_index[layer_name]
            layer = fairer_model.layers[layer_idx]
            
            if isinstance(layer, tf.keras.layers.Dense):
                mask = np.array(neuron_states, dtype=np.float32)
                weights = layer.get_weights()
                
                if len(weights) >= 2:  # weights and biases
                    # Zero out biases for dropped neurons
                    weights[1] = weights[1] * mask
                    layer.set_weights(weights)
                    
                    # Find and mask the next layer's input weights
                    for next_idx in range(layer_idx + 1, len(fairer_model.layers)):
                        next_layer = fairer_model.layers[next_idx]
                        if isinstance(next_layer, tf.keras.layers.Dense):
                            next_weights = next_layer.get_weights()
                            if len(next_weights) >= 1:
                                # Zero out input weights from dropped neurons
                                next_weights[0] = next_weights[0] * mask[:, np.newaxis]
                                next_layer.set_weights(next_weights)
                            break  # Only mask the immediate next dense layer
    
    # Save the fairer model
    fairer_model.save(output_path)
    logger.info(f"Fairer model saved to: {output_path}")
    
    # Log statistics about neuron dropping
    total_neurons = 0
    dropped_neurons = 0
    for layer_name, neuron_states in config.neuron_states.items():
        layer_total = len(neuron_states)
        layer_dropped = neuron_states.count(0)
        total_neurons += layer_total
        dropped_neurons += layer_dropped
        logger.info(f"Layer {layer_name}: {layer_dropped}/{layer_total} neurons dropped")
    
    logger.info(f"Overall: {dropped_neurons}/{total_neurons} neurons dropped "
               f"({100*dropped_neurons/total_neurons:.1f}%)")
    
    return output_path

def repair_model_fairness(original_model_path: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                         output_model_path: str, config_save_path: str = None,
                         protected_attribs: List[int] = None, max_iter: int = 10,
                         temp_schedule: str = 'linear', fairness_weight: float = 0.5) -> Tuple[str, str]:
    """
    Complete fairness repair pipeline
    
    Args:
        original_model_path: Path to original model
        X_train, y_train: Training data
        X_test, y_test: Test data
        constraint: Feature constraints for fairness evaluation
        output_model_path: Where to save the repaired model
        config_save_path: Where to save the configuration (optional)
        protected_attribs: List of protected attribute indices
        max_iter: Maximum SA iterations
        temp_schedule: Temperature schedule ('linear', 'exponential', 'logarithmic')
        fairness_weight: Weight for fairness vs accuracy tradeoff
        
    Returns:
        Tuple of (model_path, config_path)
    """
    set_all_seeds(42)
    
    # Initialize repair system
    repair_system = SimulatedAnnealingRepair(
        model_path=original_model_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        constraint=constraint,
        protected_attribs=protected_attribs,
        max_iter=max_iter,
        temp_schedule=temp_schedule,
        fairness_weight=fairness_weight
    )
    
    # Run optimization
    config = repair_system.run_sa()
    
    # Save configuration
    if config_save_path is None:
        config_save_path = output_model_path.replace('.h5', '_config.json')
    config.save(config_save_path)
    
    # Create fairer model
    create_fairer_model(original_model_path, config_save_path, output_model_path)
    
    return output_model_path, config_save_path

# Example usage
if __name__ == "__main__":
    # Example constraint for Adult dataset (13 features)
    constraint = np.array([
        [10, 100],    # age
        [0, 6],       # workclass
        [0, 15],      # education
        [1, 16],      # education-num
        [0, 6],       # marital-status
        [0, 13],      # occupation
        [0, 5],       # relationship
        [0, 4],       # race
        [0, 1],       # sex (protected attribute)
        [0, 19],      # capital-gain
        [0, 19],      # capital-loss
        [1, 100],     # hours-per-week
        [0, 40]       # native-country
    ])
    
    # Example usage (you would need to provide actual data)
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    fairer_model_path, config_path = repair_model_fairness(
        original_model_path='./neufair/model/AC-1.h5',
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        constraint=constraint,
        output_model_path='./neufair/model/AC-1-Neufair.h5',
        protected_attribs=[8],  
        max_iter=25,
        fairness_weight=0.6
    )
    
    print("NeuFair TensorFlow system ready!")
    print("Use repair_model_fairness() to repair a model's fairness issues.")
    print("Use create_fairer_model() to create a model from existing configuration.")
    
    """
    # Load your data
    X_train, y_train, X_test, y_test = load_your_data()
    
    # Repair model fairness
    fairer_model_path, config_path = repair_model_fairness(
        original_model_path='original_model.h5',
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        constraint=constraint,
        output_model_path='fairer_model.h5',
        protected_attribs=[8],  # sex attribute
        max_iter=500,
        fairness_weight=0.6
    )
    
    # Use the fairer model normally
    fairer_model = tf.keras.models.load_model(fairer_model_path)
    predictions = fairer_model.predict(X_test)
    """