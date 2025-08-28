#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import json
import os
import random
import math
import gc
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

class BalancedFastSimulatedAnnealingRepair:
    """FIXED and BALANCED TensorFlow/Keras implementation of NeuFair"""
    
    def __init__(self, model_path: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                 protected_attribs: List[int] = None, max_iter: int = 500,
                 temp_schedule: str = 'balanced', fairness_weight: float = 0.5):
        
        self.model_path = model_path
        self.original_model = tf.keras.models.load_model(model_path)
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        self.constraint = constraint
        self.protected_attribs = protected_attribs if protected_attribs else [8]
        self.max_iter = max(max_iter, 200)  # Enforce minimum iterations
        self.temp_schedule = temp_schedule
        self.fairness_weight = fairness_weight
        
        # Extract dense layers for neuron dropping
        self.dense_layers = self._get_dense_layers()
        self.layer_shapes = self._get_layer_shapes()
        
        # BALANCED: Reasonable number of fairness test samples
        self.fairness_samples = self._pregenerate_fairness_samples(n_samples=200)
        
        # Initialize state
        self.current_state = self._initialize_state()
        self.best_state = self.current_state.copy()
        self.best_cost = float('inf')
        
        # BALANCED: Smaller cache to prevent memory issues
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized BALANCED SA repair with {len(self.dense_layers)} dense layers")
        logger.info(f"Pre-generated {len(self.fairness_samples)} fairness test samples")
        logger.info(f"Minimum iterations enforced: {self.max_iter}")
    
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
    
    def _pregenerate_fairness_samples(self, n_samples: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    
    def _create_masked_model_balanced(self, state: Dict[str, List[int]]) -> tf.keras.Model:
        """BALANCED: Create masked model with efficient operations and memory management"""
        try:
            masked_model = tf.keras.models.clone_model(self.original_model)
            masked_model.set_weights(self.original_model.get_weights())
            
            # Apply masking efficiently
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
    
    def compute_fairness_balanced(self, state: Dict[str, List[int]]) -> Tuple[float, float, float]:
        """BALANCED: Compute fairness with controlled caching"""
        state_key = self._state_to_key(state)
        
        # Check cache with size control
        if state_key in self.evaluation_cache:
            cached = self.evaluation_cache[state_key]
            self.cache_hits += 1
            return cached['accuracy'], cached['f1'], cached['fairness']
        
        self.cache_misses += 1
        
        try:
            masked_model = self._create_masked_model_balanced(state)
            
            # Batch predict on test set
            y_pred_proba = masked_model.predict(self.X_test, verbose=0, batch_size=256)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = self.y_test.flatten()
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Batch fairness computation
            fairness_score = self._compute_batch_discrimination(masked_model)
            
            # Clean up model immediately
            del masked_model
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Cache results with strict size control
            if len(self.evaluation_cache) < 30:  # Reduced cache size
                self.evaluation_cache[state_key] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'fairness': fairness_score
                }
            elif len(self.evaluation_cache) >= 30:
                # Remove oldest entry
                oldest_key = next(iter(self.evaluation_cache))
                del self.evaluation_cache[oldest_key]
                self.evaluation_cache[state_key] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'fairness': fairness_score
                }
            
            return accuracy, f1, fairness_score
            
        except Exception as e:
            logger.warning(f"Error in compute_fairness_balanced: {e}")
            return 0.0, 0.0, 1.0
    
    def _compute_batch_discrimination(self, model) -> float:
        """BALANCED: Batch process discrimination detection"""
        total_discriminatory = 0
        total_samples = 0
        
        # Process in reasonable batches
        batch_size = 40
        
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
    
    def _generate_neighbor_state_balanced(self, current_state: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """FIXED: Balanced neighbor generation that actually explores neuron dropping"""
        new_state = {k: v.copy() for k, v in current_state.items()}
        
        # BALANCED: Usually flip 1 neuron, occasionally 2
        num_flips = 1 if random.random() < 0.7 else 2
        
        for _ in range(num_flips):
            layer_names = list(new_state.keys())
            selected_layer = random.choice(layer_names)
            selected_neuron = random.randint(0, len(new_state[selected_layer]) - 1)
            
            # FIXED: More balanced probabilities that actually explore dropping
            current_value = new_state[selected_layer][selected_neuron]
            
            if current_value == 1:  # Currently active
                # Drop neuron with reasonable probability
                if random.random() < 0.4:  # 40% chance to drop
                    new_state[selected_layer][selected_neuron] = 0
            else:  # Currently dropped
                # Reactivate with moderate probability
                if random.random() < 0.6:  # 60% chance to reactivate
                    new_state[selected_layer][selected_neuron] = 1
        
        return new_state
    
    def _calculate_temperature_balanced(self, iteration: int) -> float:
        """BALANCED: Temperature schedule that allows proper exploration"""
        initial_temp = 1.5
        final_temp = 0.05
        
        if self.temp_schedule == 'balanced':
            # Balanced cooling - not too fast, not too slow
            progress = iteration / self.max_iter
            return initial_temp * (final_temp / initial_temp) ** (progress ** 0.8)
        elif self.temp_schedule == 'linear':
            return initial_temp - (initial_temp - final_temp) * (iteration / self.max_iter)
        elif self.temp_schedule == 'exponential':
            return initial_temp * (final_temp / initial_temp) ** (iteration / self.max_iter)
        else:
            return initial_temp * (0.97 ** iteration)  # Moderate decay
    
    def run_sa_balanced(self) -> NeuFairConfig:
        """BALANCED: Run properly tuned simulated annealing optimization"""
        logger.info("Starting BALANCED simulated annealing optimization...")
        
        # Evaluate initial state
        accuracy, f1, fairness = self.compute_fairness_balanced(self.current_state)
        self.best_cost = self._cost_function(accuracy, f1, fairness)
        current_cost = self.best_cost
        
        logger.info(f"Initial: Accuracy={accuracy:.4f}, F1={f1:.4f}, Fairness={fairness:.4f}, Cost={self.best_cost:.4f}")
        
        # Track improvements with BALANCED early stopping
        no_improvement_count = 0
        significant_improvements = 0
        
        for iteration in range(self.max_iter):
            # Generate balanced neighbor state
            neighbor_state = self._generate_neighbor_state_balanced(self.current_state)
            
            # Evaluate neighbor
            neighbor_accuracy, neighbor_f1, neighbor_fairness = self.compute_fairness_balanced(neighbor_state)
            neighbor_cost = self._cost_function(neighbor_accuracy, neighbor_f1, neighbor_fairness)
            
            # Calculate acceptance probability
            temperature = self._calculate_temperature_balanced(iteration)
            delta_cost = neighbor_cost - current_cost
            
            if delta_cost < 0 or (temperature > 0 and random.random() < math.exp(-delta_cost / temperature)):
                self.current_state = neighbor_state
                current_cost = neighbor_cost
                
                if neighbor_cost < self.best_cost:
                    improvement = self.best_cost - neighbor_cost
                    self.best_state = {k: v.copy() for k, v in neighbor_state.items()}
                    self.best_cost = neighbor_cost
                    no_improvement_count = 0
                    significant_improvements += 1
                    
                    logger.info(f"NEW BEST at iteration {iteration}: "
                              f"Accuracy={neighbor_accuracy:.4f}, F1={neighbor_f1:.4f}, "
                              f"Fairness={neighbor_fairness:.4f}, Cost={self.best_cost:.4f} "
                              f"(improved by {improvement:.6f})")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # BALANCED: Early stopping only after sufficient exploration
            min_exploration_iterations = max(200, self.max_iter * 0.6)
            if (no_improvement_count > 150 and 
                iteration > min_exploration_iterations and 
                significant_improvements > 0):  # Only stop if we found some improvements
                logger.info(f"Early stopping at iteration {iteration} after sufficient exploration")
                logger.info(f"Total significant improvements: {significant_improvements}")
                break
            
            # Log progress
            if (iteration + 1) % 50 == 0:
                # Count dropped neurons in current best state
                total_neurons = sum(len(neurons) for neurons in self.best_state.values())
                dropped_neurons = sum(neurons.count(0) for neurons in self.best_state.values())
                drop_percentage = (dropped_neurons / total_neurons) * 100 if total_neurons > 0 else 0
                
                logger.info(f"Iteration {iteration + 1}/{self.max_iter}: "
                          f"Temp={temperature:.4f}, Current Cost={current_cost:.4f}, "
                          f"Best Cost={self.best_cost:.4f}, "
                          f"Neurons dropped: {dropped_neurons}/{total_neurons} ({drop_percentage:.1f}%), "
                          f"Cache: {len(self.evaluation_cache)} (hits/misses: {self.cache_hits}/{self.cache_misses})")
        
        # Final statistics
        total_neurons = sum(len(neurons) for neurons in self.best_state.values())
        dropped_neurons = sum(neurons.count(0) for neurons in self.best_state.values())
        drop_percentage = (dropped_neurons / total_neurons) * 100 if total_neurons > 0 else 0
        
        logger.info(f"FINAL RESULT: {dropped_neurons}/{total_neurons} neurons dropped ({drop_percentage:.1f}%)")
        logger.info(f"Total significant improvements: {significant_improvements}")
        logger.info(f"Cache efficiency: {self.cache_hits}/{self.cache_hits + self.cache_misses} hits")
        
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
        
        logger.info("BALANCED simulated annealing completed!")
        return config

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

def repair_model_fairness_balanced(original_model_path: str, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray, constraint: np.ndarray,
                                  output_model_path: str, config_save_path: str = None,
                                  protected_attribs: List[int] = None, max_iter: int = 300,
                                  temp_schedule: str = 'balanced', fairness_weight: float = 0.5) -> Tuple[str, str]:
    """
    BALANCED Complete fairness repair pipeline
    
    FIXES APPLIED:
    - Balanced neighbor generation (40% drop chance, 60% reactivation chance)
    - Minimum 200 iterations enforced
    - Proper exploration before early stopping (60% of max_iter minimum)
    - Better temperature schedule
    - Controlled memory usage with smaller cache
    - Detailed logging of neuron dropping progress
    """
    set_all_seeds(42)
    
    logger.info("=== STARTING BALANCED NEUFAIR REPAIR ===")
    
    # Initialize repair system with balanced implementation
    repair_system = BalancedFastSimulatedAnnealingRepair(
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
    config = repair_system.run_sa_balanced()
    
    # Save configuration
    if config_save_path is None:
        config_save_path = output_model_path.replace('.h5', '_config.json')
    config.save(config_save_path)
    
    # Create fairer model
    create_fairer_model(original_model_path, config_save_path, output_model_path)
    
    return output_model_path, config_save_path

# Example usage with balanced parameters
if __name__ == "__main__":
    from preprocess import load_adult_ac1
    
    constraint = np.array([
        [10, 100], [0, 6], [0, 15], [1, 16], [0, 6], [0, 13], [0, 5], 
        [0, 4], [0, 1], [0, 19], [0, 19], [1, 100], [0, 40]
    ])
    
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    print("=== RUNNING BALANCED VERSION ===")
    fairer_model_path, config_path = repair_model_fairness_balanced(
        original_model_path='./neufair/model/AC-5.h5',
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        constraint=constraint, output_model_path='./neufair/model/AC-5-Neufair.h5',
        protected_attribs=[8], max_iter=300, fairness_weight=0.6
    )
    
    print("=== BALANCED NEUFAIR COMPLETED ===")
    print(f"Fairer model: {fairer_model_path}")
    print(f"Config: {config_path}")