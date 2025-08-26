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

class FastSimulatedAnnealingRepair:
    """OPTIMIZED TensorFlow/Keras implementation of NeuFair"""
    
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
        
        # OPTIMIZATION: Pre-generate fairness test samples
        self.fairness_samples = self._pregenerate_fairness_samples(n_samples=100)  # Reduced from 500
        
        # Initialize state
        self.current_state = self._initialize_state()
        self.best_state = self.current_state.copy()
        self.best_cost = float('inf')
        
        # OPTIMIZATION: Cache for model evaluations
        self.evaluation_cache = {}
        
        logger.info(f"Initialized FAST SA repair with {len(self.dense_layers)} dense layers")
        logger.info(f"Pre-generated {len(self.fairness_samples)} fairness test samples")
    
    def _get_dense_layers(self) -> List[tf.keras.layers.Dense]:
        """Extract all dense layers from the model"""
        dense_layers = []
        for layer in self.original_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                if layer != self.original_model.layers[-1]:
                    dense_layers.append(layer)
        return dense_layers
    
    def _get_layer_shapes(self) -> List[int]:
        """Get the number of neurons in each dense layer"""
        return [layer.units for layer in self.dense_layers]
    
    def _initialize_state(self) -> Dict[str, List[int]]:
        """Initialize the neuron dropping state (all neurons active)"""
        state = {}
        for layer in self.dense_layers:
            state[layer.name] = [1] * layer.units
        return state
    
    def _pregenerate_fairness_samples(self, n_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Pre-generate samples for fairness evaluation"""
        samples = []
        for _ in range(n_samples):
            # Generate random instance
            x = np.array([
                np.random.randint(self.constraint[i][0], self.constraint[i][1] + 1) 
                for i in range(len(self.constraint))
            ]).astype(np.float32)
            
            # Create similar instances by varying protected attributes
            similar_instances = self._generate_similar_instances(x)
            samples.append((x, similar_instances))
        
        return samples
    
    def _state_to_key(self, state: Dict[str, List[int]]) -> str:
        """Convert state to hashable key for caching"""
        return json.dumps({k: v for k, v in sorted(state.items())})
    
    def _create_masked_model_fast(self, state: Dict[str, List[int]]) -> tf.keras.Model:
        """OPTIMIZED: Create masked model with minimal operations"""
        # Check cache first
        state_key = self._state_to_key(state)
        if state_key in self.evaluation_cache:
            return self.evaluation_cache[state_key]['model']
        
        try:
            masked_model = tf.keras.models.clone_model(self.original_model)
            masked_model.set_weights(self.original_model.get_weights())
            
            # Apply masking more efficiently
            self._apply_neuron_masks_vectorized(masked_model, state)
            
            return masked_model
            
        except Exception as e:
            logger.error(f"Error creating masked model: {e}")
            raise
    
    def _apply_neuron_masks_vectorized(self, model: tf.keras.Model, state: Dict[str, List[int]]):
        """Apply neuron masks using vectorized operations"""
        layer_name_to_index = {layer.name: idx for idx, layer in enumerate(model.layers)}
        
        for layer_name, neuron_states in state.items():
            if layer_name in layer_name_to_index:
                layer_idx = layer_name_to_index[layer_name]
                layer = model.layers[layer_idx]
                
                if isinstance(layer, tf.keras.layers.Dense):
                    mask = np.array(neuron_states, dtype=np.float32)
                    weights = layer.get_weights()
                    
                    if len(weights) >= 2:
                        # Vectorized bias masking
                        weights[1] = weights[1] * mask
                        layer.set_weights(weights)
                        
                        # Vectorized next layer weight masking
                        for next_idx in range(layer_idx + 1, len(model.layers)):
                            next_layer = model.layers[next_idx]
                            if isinstance(next_layer, tf.keras.layers.Dense):
                                next_weights = next_layer.get_weights()
                                if len(next_weights) >= 1:
                                    next_weights[0] = next_weights[0] * mask[:, np.newaxis]
                                    next_layer.set_weights(next_weights)
                                break
    
    def compute_fairness_fast(self, state: Dict[str, List[int]]) -> Tuple[float, float, float]:
        """OPTIMIZED: Batch process fairness computation"""
        state_key = self._state_to_key(state)
        
        # Check cache
        if state_key in self.evaluation_cache:
            cached = self.evaluation_cache[state_key]
            return cached['accuracy'], cached['f1'], cached['fairness']
        
        try:
            masked_model = self._create_masked_model_fast(state)
            
            # OPTIMIZATION: Batch predict on test set
            y_pred_proba = masked_model.predict(self.X_test, verbose=0, batch_size=512)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = self.y_test.flatten()
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # OPTIMIZATION: Batch fairness computation
            fairness_score = self._compute_batch_discrimination(masked_model)
            
            # Cache results
            self.evaluation_cache[state_key] = {
                'model': masked_model,
                'accuracy': accuracy,
                'f1': f1,
                'fairness': fairness_score
            }
            
            # Limit cache size
            if len(self.evaluation_cache) > 50:
                oldest_key = next(iter(self.evaluation_cache))
                del self.evaluation_cache[oldest_key]['model']
                del self.evaluation_cache[oldest_key]
            
            return accuracy, f1, fairness_score
            
        except Exception as e:
            logger.warning(f"Error in compute_fairness_fast: {e}")
            return 0.0, 0.0, 1.0
    
    def _compute_batch_discrimination(self, model) -> float:
        """OPTIMIZED: Batch process discrimination detection"""
        total_discriminatory = 0
        total_samples = 0
        
        # Process in batches for efficiency
        batch_size = 25  # Reduced from processing 500 samples
        
        for i in range(0, len(self.fairness_samples), batch_size):
            batch_samples = self.fairness_samples[i:i+batch_size]
            
            # Batch process original instances
            original_instances = np.array([sample[0] for sample in batch_samples])
            original_preds = (model.predict(original_instances, verbose=0, batch_size=len(original_instances)) > 0.5).astype(int)
            
            # Check each sample's similar instances
            for j, (original_x, similar_instances) in enumerate(batch_samples):
                original_pred = original_preds[j][0]
                
                # Batch predict similar instances
                similar_preds = (model.predict(similar_instances, verbose=0, batch_size=len(similar_instances)) > 0.5).astype(int)
                
                # Check if any similar instance has different prediction
                if np.any(similar_preds.flatten() != original_pred):
                    total_discriminatory += 1
                
                total_samples += 1
        
        return total_discriminatory / total_samples if total_samples > 0 else 0.0
    
    def _generate_similar_instances(self, x: np.ndarray) -> np.ndarray:
        """Generate similar instances by varying protected attributes"""
        similar_instances = []
        
        # OPTIMIZATION: Limit combinations for speed
        for attr_idx in self.protected_attribs:
            min_val, max_val = self.constraint[attr_idx]
            for value in range(min_val, max_val + 1):
                if value != x[attr_idx]:  # Skip the original value
                    x_new = x.copy()
                    x_new[attr_idx] = value
                    similar_instances.append(x_new)
        
        return np.array(similar_instances) if similar_instances else np.array([x])
    
    def _cost_function(self, accuracy: float, f1: float, fairness_score: float) -> float:
        """Compute the cost function balancing accuracy and fairness"""
        performance_score = (accuracy + f1) / 2
        cost = self.fairness_weight * fairness_score + (1 - self.fairness_weight) * (1 - performance_score)
        return cost
    
    def _generate_neighbor_state_smart(self, current_state: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """OPTIMIZED: Smart neighbor generation with multiple flips"""
        new_state = {k: v.copy() for k, v in current_state.items()}
        
        # OPTIMIZATION: Flip 2-3 neurons at once for bigger moves
        num_flips = random.choice([1, 2, 3])
        
        for _ in range(num_flips):
            layer_names = list(new_state.keys())
            selected_layer = random.choice(layer_names)
            selected_neuron = random.randint(0, len(new_state[selected_layer]) - 1)
            
            # Flip with bias towards keeping neurons active
            if new_state[selected_layer][selected_neuron] == 1:
                if random.random() < 0.2:  # Reduced drop probability
                    new_state[selected_layer][selected_neuron] = 0
            else:
                if random.random() < 0.8:  # High reactivation probability
                    new_state[selected_layer][selected_neuron] = 1
        
        return new_state
    
    def _calculate_temperature(self, iteration: int) -> float:
        """Calculate temperature for simulated annealing"""
        initial_temp = 2.0  # Higher initial temp for faster exploration
        final_temp = 0.01
        
        if self.temp_schedule == 'fast_linear':
            # Faster cooling schedule
            return initial_temp - (initial_temp - final_temp) * ((iteration / self.max_iter) ** 0.5)
        elif self.temp_schedule == 'exponential':
            return initial_temp * (final_temp / initial_temp) ** (iteration / self.max_iter)
        else:
            return initial_temp * (0.98 ** iteration)  # Faster decay
    
    def run_sa_fast(self) -> NeuFairConfig:
        """OPTIMIZED: Run fast simulated annealing optimization"""
        logger.info("Starting FAST simulated annealing optimization...")
        
        # Evaluate initial state
        accuracy, f1, fairness = self.compute_fairness_fast(self.current_state)
        self.best_cost = self._cost_function(accuracy, f1, fairness)
        current_cost = self.best_cost
        
        logger.info(f"Initial: Accuracy={accuracy:.4f}, F1={f1:.4f}, Fairness={fairness:.4f}, Cost={self.best_cost:.4f}")
        
        # OPTIMIZATION: Track recent improvements for early stopping
        no_improvement_count = 0
        last_best_cost = self.best_cost
        
        for iteration in range(self.max_iter):
            # Generate neighbor state with smart heuristics
            neighbor_state = self._generate_neighbor_state_smart(self.current_state)
            
            # Evaluate neighbor
            neighbor_accuracy, neighbor_f1, neighbor_fairness = self.compute_fairness_fast(neighbor_state)
            neighbor_cost = self._cost_function(neighbor_accuracy, neighbor_f1, neighbor_fairness)
            
            # Calculate acceptance probability
            temperature = self._calculate_temperature(iteration)
            delta_cost = neighbor_cost - current_cost
            
            if delta_cost < 0 or (temperature > 0 and random.random() < math.exp(-delta_cost / temperature)):
                self.current_state = neighbor_state
                current_cost = neighbor_cost
                
                if neighbor_cost < self.best_cost:
                    self.best_state = {k: v.copy() for k, v in neighbor_state.items()}
                    improvement = self.best_cost - neighbor_cost
                    self.best_cost = neighbor_cost
                    no_improvement_count = 0
                    logger.info(f"NEW BEST at iteration {iteration}: "
                              f"Accuracy={neighbor_accuracy:.4f}, F1={neighbor_f1:.4f}, "
                              f"Fairness={neighbor_fairness:.4f}, Cost={self.best_cost:.4f} "
                              f"(improved by {improvement:.6f})")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # OPTIMIZATION: Early stopping if no improvement
            if no_improvement_count > 50 and iteration > self.max_iter * 0.3:
                logger.info(f"Early stopping at iteration {iteration} (no improvement for 50 iterations)")
                break
            
            # Log progress less frequently
            if (iteration + 1) % 50 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iter}: "
                          f"Temp={temperature:.4f}, Current Cost={current_cost:.4f}, "
                          f"Best Cost={self.best_cost:.4f}, Cache size={len(self.evaluation_cache)}")
        
        # Create configuration
        model_architecture = {
            'layer_names': [layer.name for layer in self.dense_layers],
            'layer_shapes': self.layer_shapes,
            'total_layers': len(self.dense_layers)
        }
        
        config = NeuFairConfig(
            model_architecture=model_architecture,
            neuron_states=self.best_state
        )
        
        logger.info("FAST simulated annealing completed!")
        return config

def repair_model_fairness_fast(original_model_path: str, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                              output_model_path: str, config_save_path: str = None,
                              protected_attribs: List[int] = None, max_iter: int = 100,  # Reduced default
                              temp_schedule: str = 'fast_linear', fairness_weight: float = 0.5) -> Tuple[str, str]:
    """
    FAST Complete fairness repair pipeline
    
    OPTIMIZATIONS APPLIED:
    - Reduced default iterations from 1000 to 100
    - Batch processing for predictions
    - Caching of model evaluations
    - Reduced fairness samples from 500 to 100
    - Smart neighbor generation (multiple neuron flips)
    - Early stopping when no improvement
    - Fast temperature schedule
    """
    set_all_seeds(42)
    
    logger.info("=== STARTING FAST NEUFAIR REPAIR ===")
    
    # Initialize repair system with fast implementation
    repair_system = FastSimulatedAnnealingRepair(
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
    config = repair_system.run_sa_fast()
    
    # Save configuration
    if config_save_path is None:
        config_save_path = output_model_path.replace('.h5', '_config.json')
    config.save(config_save_path)
    
    # Create fairer model (reuse from original code)
    create_fairer_model(original_model_path, config_save_path, output_model_path)
    
    return output_model_path, config_save_path

# ULTRA FAST version for quick testing
def repair_model_fairness_ultra_fast(original_model_path: str, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                                    output_model_path: str, **kwargs) -> Tuple[str, str]:
    """ULTRA FAST version - 5-10 minutes instead of hours"""
    return repair_model_fairness_fast(
        original_model_path=original_model_path,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        constraint=constraint, output_model_path=output_model_path,
        max_iter=50,  # Very low iterations
        **kwargs
    )

# Include original functions for compatibility
def create_fairer_model(original_model_path: str, config_path: str, output_path: str) -> str:
    """Create a fairer model by permanently applying neuron dropping configuration"""
    logger.info("Creating fairer model from configuration...")
    
    original_model = tf.keras.models.load_model(original_model_path)
    config = NeuFairConfig.load(config_path)
    
    try:
        fairer_model = tf.keras.models.clone_model(original_model)
        fairer_model.set_weights(original_model.get_weights())
    except Exception as e:
        logger.error(f"Error cloning model: {e}")
        raise
    
    # Apply permanent neuron masking
    layer_name_to_index = {layer.name: idx for idx, layer in enumerate(fairer_model.layers)}
    
    for layer_name, neuron_states in config.neuron_states.items():
        if layer_name in layer_name_to_index:
            layer_idx = layer_name_to_index[layer_name]
            layer = fairer_model.layers[layer_idx]
            
            if isinstance(layer, tf.keras.layers.Dense):
                mask = np.array(neuron_states, dtype=np.float32)
                weights = layer.get_weights()
                
                if len(weights) >= 2:
                    weights[1] = weights[1] * mask
                    layer.set_weights(weights)
                    
                    for next_idx in range(layer_idx + 1, len(fairer_model.layers)):
                        next_layer = fairer_model.layers[next_idx]
                        if isinstance(next_layer, tf.keras.layers.Dense):
                            next_weights = next_layer.get_weights()
                            if len(next_weights) >= 1:
                                next_weights[0] = next_weights[0] * mask[:, np.newaxis]
                                next_layer.set_weights(next_weights)
                            break
    
    fairer_model.save(output_path)
    logger.info(f"Fairer model saved to: {output_path}")
    
    # Log statistics
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

# Example usage optimized for speed
if __name__ == "__main__":
    from preprocess import load_adult_ac1
    
    constraint = np.array([
        [10, 100], [0, 6], [0, 15], [1, 16], [0, 6], [0, 13], [0, 5], 
        [0, 4], [0, 1], [0, 19], [0, 19], [1, 100], [0, 40]
    ])
    
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    print("=== RUNNING ULTRA FAST VERSION (5-10 minutes) ===")
    fairer_model_path, config_path = repair_model_fairness_ultra_fast(
        original_model_path='./neufair/model/AC-1.h5',
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        constraint=constraint, output_model_path='./neufair/model/AC-1-Neufair-Fast.h5',
        protected_attribs=[8], fairness_weight=0.6
    )
    
    print("=== FAST NEUFAIR COMPLETED ===")
    print(f"Fairer model: {fairer_model_path}")
    print(f"Config: {config_path}")