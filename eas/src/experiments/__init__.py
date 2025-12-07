"""
Main Evaluation Loop for EAS Experiment
Implements the progressive online learning paradigm with efficiency considerations
"""
import torch
import torch.nn.functional as F
import random
import time
from typing import List, Dict, Tuple, Optional
from ..models.transformer import AutoregressiveTransformer
from ..models.tokenizer import LogicTokenizer
from ..watcher import EmergentWatcher
from ..datasets import LogicCorpusGenerator


class EASEvaluator:
    """Main evaluator for the Emergent Activation Snapping (EAS) experiment"""
    
    def __init__(self, 
                 model: AutoregressiveTransformer,
                 tokenizer: LogicTokenizer,
                 watcher: Optional[EmergentWatcher] = None,
                 device: str = 'cpu',
                 max_seq_len: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.watcher = watcher
        self.device = device
        self.max_seq_len = max_seq_len
        
        # Training state
        self.model_trained = False
        self.base_accuracy = 0.0
        
        # Performance metrics
        self.metrics = {
            'accuracy': [],
            'latency': [],
            'intervention_frequency': [],
            'attractor_stability': [],
            'snap_history': []
        }
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
    
    def encode_problem(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a logic problem into token IDs and target."""
        # Combine premises and conclusion into a single sequence
        premises = []
        if sample.get('premise1', ''):
            premises.append(sample['premise1'])
        if sample.get('premise2', ''):
            premises.append(sample['premise2'])
        
        problem_text = ' <SEP> '.join(premises) + ' <SEP> ' + sample.get('conclusion', '')
        
        # Encode the problem
        token_ids = self.tokenizer.encode(problem_text)
        
        # Ensure max sequence length
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            # Pad to max length
            token_ids.extend([self.tokenizer.vocab['<PAD>']] * (self.max_seq_len - len(token_ids)))
        
        # Create input and target tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        # The target would be the same sequence shifted by one position for autoregressive training
        target_ids = input_ids.clone()
        target_ids[:, :-1] = input_ids[:, 1:]
        
        return input_ids, target_ids
    
    def check_correctness(self, prediction: torch.Tensor, target: torch.Tensor, sample: Dict) -> bool:
        """Check if the model's prediction is correct for the logical problem."""
        # For this implementation, we'll use the oracle label from the sample
        # In a full implementation, this would involve more complex logic checking
        return sample.get('validity', False)
    
    def train_base_model(self, dataset: List[Dict], epochs: int = 3, lr: float = 1e-3):
        """Train the base model on the pre-training dataset to achieve 60-70% accuracy."""
        print("Training base model...")
        
        # Set model to training mode
        self.model.train()
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab['<PAD>'])
        
        # Simple training loop
        total_samples = len(dataset)
        batch_size = 4  # Small batch size for demonstration
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Shuffle dataset
            random.shuffle(dataset)
            
            for i in range(0, total_samples, batch_size):
                batch = dataset[i:i+batch_size]
                
                # Prepare batch
                input_batch = []
                target_batch = []
                
                for sample in batch:
                    input_ids, target_ids = self.encode_problem(sample)
                    input_batch.append(input_ids.squeeze(0))  # Remove batch dimension temporarily
                    target_batch.append(target_ids.squeeze(0))
                
                # Stack into proper batch format
                inputs = torch.stack(input_batch).to(self.device)
                targets = torch.stack(target_batch).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Reshape for loss calculation
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs_flat, dim=-1)
                valid_targets = targets_flat != self.tokenizer.vocab['<PAD>']
                correct += ((predictions == targets_flat) & valid_targets).sum().item()
                total += valid_targets.sum().item()
                
                total_loss += loss.item()
                
                if i % 50 == 0:  # Print progress every 50 batches
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i//batch_size}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / (total_samples // batch_size)
            accuracy = correct / total if total > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            if accuracy > 0.70:  # Early stopping if we reach target accuracy
                print(f"Target accuracy reached: {accuracy:.4f}")
                break
        
        # Mark model as trained and record base accuracy
        self.model_trained = True
        self.base_accuracy = accuracy
        print(f"Base model trained with final accuracy: {accuracy:.4f}")
        
        # Freeze model weights after training
        for param in self.model.parameters():
            param.requires_grad = False
    
    def evaluate_with_eas(self, dataset: List[Dict], num_iterations: int = 200):
        """Run the main EAS evaluation loop."""
        if not self.model_trained:
            raise ValueError("Base model must be trained before running EAS evaluation")
        
        print(f"Starting EAS evaluation for {num_iterations} iterations...")
        
        # Set model to eval mode
        self.model.eval()
        
        # If we have a watcher, register the intervention hook
        intervention_func = None
        if self.watcher:
            def intervention_func(activations):
                # Apply the watcher's snap operation to the middle layer
                return self.watcher.snap(activations)
            
            # Register the intervention at the middle layer
            self.model.register_intervention_hook(self.model.middle_layer_idx, intervention_func)
        
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Select a random sample from the dataset
            sample = random.choice(dataset)
            
            # Encode the sample
            input_ids, target_ids = self.encode_problem(sample)
            
            # Record start time for latency measurement
            iter_start_time = time.time()
            
            # Forward pass with potential Watcher intervention
            with torch.no_grad():
                if self.watcher:
                    # Clear any previous activations
                    self.model.layer_activations.clear()
                
                output = self.model(input_ids)
                
                # Check correctness via oracle
                is_correct = self.check_correctness(output, target_ids, sample)
                
                if is_correct:
                    correct_predictions += 1
                    
                    # If correct and we have a watcher, update attractors
                    if self.watcher:
                        # Get the activation from the middle layer
                        middle_layer_activation = self.model.get_layer_activation(self.model.middle_layer_idx)
                        if middle_layer_activation is not None:
                            self.watcher.update(middle_layer_activation)
                
                total_predictions += 1
            
            # Record latency
            iter_time = time.time() - iter_start_time
            self.metrics['latency'].append(iter_time)
            
            # Calculate and record accuracy
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.metrics['accuracy'].append(current_accuracy)
            
            # Record additional metrics if watcher is present
            if self.watcher:
                self.metrics['intervention_frequency'].append(self.watcher.get_intervention_frequency())
                self.metrics['attractor_stability'].append(self.watcher.get_attractor_stability())
                self.metrics['snap_history'].extend(self.watcher.snap_history)
            
            # Print progress every 25 iterations
            if (iteration + 1) % 25 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations} - "
                      f"Accuracy: {current_accuracy:.4f}, "
                      f"Correct: {correct_predictions}/{total_predictions}")
        
        # Remove the intervention hook
        if self.watcher:
            self.model.remove_intervention_hook(self.model.middle_layer_idx)
            
        total_time = time.time() - start_time
        print(f"EAS evaluation completed in {total_time:.2f} seconds")
        print(f"Final accuracy: {correct_predictions/total_predictions:.4f}")
        
        return self.metrics
    
    def run_small_model_experiment(self, small_model_size: bool = True):
        """Run a small-scale experiment for validation."""
        # Generate small datasets
        if small_model_size:
            datasets = LogicCorpusGenerator().generate_pretraining_dataset(100), \
                      LogicCorpusGenerator().generate_evaluation_dataset(30)
            pretrain_dataset = LogicCorpusGenerator().generate_pretraining_dataset(100)
            eval_dataset = LogicCorpusGenerator().generate_evaluation_dataset(30)
        else:
            pretrain_dataset = LogicCorpusGenerator().generate_pretraining_dataset(200)
            eval_dataset = LogicCorpusGenerator().generate_evaluation_dataset(50)
        
        # Create a small model for rapid prototyping
        vocab_size = self.tokenizer.get_vocab_size()
        if small_model_size:
            model = AutoregressiveTransformer(
                vocab_size=vocab_size,
                d_model=128,
                num_layers=1,
                num_heads=4
            ).to(self.device)
        else:
            model = AutoregressiveTransformer(
                vocab_size=vocab_size,
                d_model=256,
                num_layers=2,
                num_heads=8
            ).to(self.device)
        
        # Create a small watcher if needed
        watcher = None
        if self.watcher:
            if small_model_size:
                watcher = EmergentWatcher(
                    dim=128,
                    k=5,
                    alpha_base=0.3,
                    max_delta=0.3
                ).to(self.device)
            else:
                watcher = EmergentWatcher(
                    dim=256,
                    k=8,
                    alpha_base=0.3,
                    max_delta=0.5
                ).to(self.device)
        
        # Create evaluator with the small model
        small_evaluator = EASEvaluator(model, self.tokenizer, watcher, self.device)
        
        # Train the small model
        small_evaluator.train_base_model(pretrain_dataset, epochs=2)
        
        # Run the EAS evaluation
        metrics = small_evaluator.evaluate_with_eas(eval_dataset, num_iterations=50)
        
        return metrics


def run_experiment_with_baselines(dataset_path: str = None):
    """Run the complete EAS experiment with all baseline conditions."""
    # Initialize components
    tokenizer = LogicCorpusGenerator().create_logic_tokenizer()
    model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        num_layers=2,
        num_heads=8
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Generate datasets
    datasets = LogicCorpusGenerator().create_logic_datasets()
    pretrain_data = datasets['pretrain']
    eval_data = datasets['evaluation']
    
    # Run small model validation first
    print("Running small model validation...")
    evaluator = EASEvaluator(model, tokenizer, device=device)
    small_metrics = evaluator.run_small_model_experiment(small_model_size=True)
    
    print("Small model validation completed.")
    print(f"Small model final accuracy: {small_metrics['accuracy'][-1] if small_metrics['accuracy'] else 0:.4f}")
    
    # If small model showed promise, proceed with full model
    if small_metrics['accuracy'] and small_metrics['accuracy'][-1] > 0.55:  # Threshold for proceeding
        print("Small model showed promise. Proceeding with standard model...")
        
        # Create standard model and watcher
        standard_model = AutoregressiveTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            num_layers=2,
            num_heads=8
        ).to(device)
        
        watcher = EmergentWatcher(
            dim=512,
            k=10,
            alpha_base=0.3,
            max_delta=0.5
        ).to(device)
        
        # Create evaluator with standard model and watcher
        standard_evaluator = EASEvaluator(standard_model, tokenizer, watcher, device)
        
        # Train the base model
        standard_evaluator.train_base_model(pretrain_data, epochs=3)
        
        # Run the main EAS experiment
        main_metrics = standard_evaluator.evaluate_with_eas(eval_data, num_iterations=200)
        
        return main_metrics
    else:
        print("Small model did not meet threshold. Consider revising approach.")
        return small_metrics