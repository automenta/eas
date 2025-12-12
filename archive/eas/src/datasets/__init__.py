"""
Synthetic Logic Corpus Generator for EAS
Generates syllogisms and propositional logic samples for training and evaluation
"""
import json
import random
from typing import List, Dict, Tuple
from enum import Enum


class LogicalType(Enum):
    """Types of logical reasoning problems"""
    SYLLOGISM_CLASSIC = "syllogism_classic"
    PROPOSITIONAL_LOGIC = "propositional_logic"
    TRANSITIVITY = "transitivity"
    NEGATION = "negation"
    CONJUNCTION = "conjunction"


class LogicCorpusGenerator:
    """Generator for synthetic logic reasoning datasets"""
    
    def __init__(self):
        # Subjects for syllogisms
        self.subjects = [
            "A", "B", "C", "X", "Y", "Z", "S", "P", "Q", "R", 
            "man", "woman", "person", "student", "teacher", 
            "dog", "cat", "animal", "mammal", "bird"
        ]
        
        # Predicates for syllogisms
        self.predicates = [
            "mortal", "mortal_being", "living_thing", "animal", 
            "rational", "wise", "student", "teacher", "smart",
            "mammal", "feathered", "flying", "vertebrate"
        ]
        
        # Quantifiers for syllogisms
        self.quantifiers = [
            "all", "some", "no", "every", "each", "any"
        ]
        
        # Propositional variables
        self.propositions = ["P", "Q", "R", "S", "T"]
        
        # Connectives for propositional logic
        self.connectives = [
            "and", "or", "if_then", "iff", "not"
        ]

    def generate_classic_syllogism(self) -> Tuple[str, bool]:
        """Generate a classic syllogism: All X are Y. Z is X. -> Z is Y."""
        subject1 = random.choice(self.subjects)
        predicate1 = random.choice(self.predicates)
        subject2 = random.choice(self.subjects)
        
        premise1 = f"all {subject1} are {predicate1}"
        premise2 = f"{subject2} is {subject1}"
        conclusion = f"{subject2} is {predicate1}"
        
        # Valid syllogism
        problem = f"{premise1}. {premise2}. -> {conclusion}"
        return problem, True

    def generate_invalid_syllogism(self) -> Tuple[str, bool]:
        """Generate an invalid syllogism for training."""
        subject1 = random.choice(self.subjects)
        predicate1 = random.choice(self.predicates)
        subject2 = random.choice(self.subjects)
        predicate2 = random.choice(self.predicates)
        
        premise1 = f"all {subject1} are {predicate1}"
        premise2 = f"{subject2} is {predicate2}"
        conclusion = f"{subject2} is {predicate1}"
        
        # Invalid syllogism (unless by coincidence subject2 is also subject1)
        problem = f"{premise1}. {premise2}. -> {conclusion}"
        return problem, False

    def generate_propositional_logic(self) -> Tuple[str, bool]:
        """Generate propositional logic: If P then Q. P. -> Q."""
        prop1, prop2 = random.sample(self.propositions, 2)
        
        premise1 = f"if {prop1} then {prop2}"
        premise2 = f"{prop1}"
        conclusion = f"{prop2}"
        
        # Valid modus ponens
        problem = f"{premise1}. {prop2}. -> {conclusion}"
        return problem, True

    def generate_negation_logic(self) -> Tuple[str, bool]:
        """Generate logic involving negation."""
        subject = random.choice(self.subjects)
        predicate = random.choice(self.predicates)
        
        premise1 = f"no {subject} is {predicate}"
        premise2 = f"X is {predicate}"
        conclusion = f"X is not {subject}"
        
        problem = f"{premise1}. {premise2}. -> {conclusion}"
        return problem, True

    def generate_conjunction_logic(self) -> Tuple[str, bool]:
        """Generate logic involving conjunctions."""
        prop1, prop2, prop3 = random.sample(self.propositions, 3)
        
        premise1 = f"{prop1} and {prop2}"
        premise2 = f"if {prop1} and {prop2} then {prop3}"
        conclusion = f"{prop3}"
        
        problem = f"{premise1}. {premise2}. -> {conclusion}"
        return problem, True

    def generate_transitivity_logic(self) -> Tuple[str, bool]:
        """Generate logic involving transitivity."""
        subject1, subject2, subject3 = random.sample(self.subjects, 3)
        predicate = random.choice(self.predicates)

        premise1 = f"{subject1} is the same as {subject2}"
        premise2 = f"{subject2} is the same as {subject3}"
        conclusion = f"{subject1} is the same as {subject3}"

        problem = f"{premise1}. {premise2}. -> {conclusion}"
        return problem, True

    def generate_sample(self, logical_type: LogicalType = None) -> Dict:
        """Generate a single logic sample with structured format."""
        if logical_type is None:
            logical_type = random.choice(list(LogicalType))

        if logical_type == LogicalType.SYLLOGISM_CLASSIC:
            problem, is_valid = self.generate_classic_syllogism()
        elif logical_type == LogicalType.PROPOSITIONAL_LOGIC:
            problem, is_valid = self.generate_propositional_logic()
        elif logical_type == LogicalType.NEGATION:
            problem, is_valid = self.generate_negation_logic()
        elif logical_type == LogicalType.CONJUNCTION:
            problem, is_valid = self.generate_conjunction_logic()
        elif logical_type == LogicalType.TRANSITIVITY:
            problem, is_valid = self.generate_transitivity_logic()
        else:
            # Default to classic syllogism
            problem, is_valid = self.generate_classic_syllogism()

        # Parse the problem into premises and conclusion
        parts = problem.split(" -> ")
        if len(parts) == 2:
            premises_text = parts[0].strip()
            conclusion = parts[1].strip()

            # Split premises
            premises = [p.strip() for p in premises_text.split(".") if p.strip()]
        else:
            premises = []
            conclusion = problem

        return {
            "premise1": premises[0] if len(premises) > 0 else "",
            "premise2": premises[1] if len(premises) > 1 else "",
            "conclusion": conclusion,
            "validity": is_valid,
            "logical_type": logical_type.value,
            "problem_text": problem
        }

    def generate_challenging_sample(self) -> Dict:
        """Generate a more challenging sample that might be harder for the model to solve."""
        # Introduce more subtle logical errors or complex reasoning
        # This will create problems where EAS might provide more benefit

        # Option 1: Valid syllogism
        if random.random() < 0.5:
            return self.generate_sample()
        else:
            # Option 2: Invalid syllogism (more challenging as model needs to identify invalidity)
            subject1, subject2, subject3 = random.sample(self.subjects[:10], 3)  # Use first 10 subjects
            predicate1, predicate2 = random.sample(self.predicates[:10], 2)  # Use first 10 predicates

            # Create a logically invalid syllogism
            premise1 = f"all {subject1} are {predicate1}"
            premise2 = f"all {subject2} are {predicate1}"  # This doesn't connect properly
            conclusion = f"all {subject2} are {subject1}"  # Invalid conclusion

            problem_text = f"{premise1}. {premise2}. -> {conclusion}"

            # Parse the problem
            parts = problem_text.split(" -> ")
            if len(parts) == 2:
                premises_text = parts[0].strip()
                conclusion = parts[1].strip()
                premises = [p.strip() for p in premises_text.split(".") if p.strip()]
            else:
                premises = []
                conclusion = problem_text

            return {
                "premise1": premises[0] if len(premises) > 0 else "",
                "premise2": premises[1] if len(premises) > 1 else "",
                "conclusion": conclusion,
                "validity": False,  # This is an invalid syllogism
                "logical_type": LogicalType.SYLLOGISM_CLASSIC.value,
                "problem_text": problem_text
            }

    def generate_dataset(self, size: int, include_invalid: bool = True) -> List[Dict]:
        """Generate a dataset of logic samples."""
        dataset = []

        for i in range(size):
            # Use the more challenging sample generator
            sample = self.generate_challenging_sample()
            dataset.append(sample)

        return dataset

    def generate_pretraining_dataset(self, size: int = 1000) -> List[Dict]:
        """Generate a dataset for pre-training the base model."""
        return self.generate_dataset(size, include_invalid=True)

    def generate_evaluation_dataset(self, size: int = 200) -> List[Dict]:
        """Generate a balanced dataset for online evaluation."""
        dataset = []
        
        # Ensure balanced distribution of logical types
        samples_per_type = size // len(LogicalType)
        
        for logical_type in LogicalType:
            for _ in range(samples_per_type):
                sample = self.generate_sample(logical_type)
                dataset.append(sample)
        
        # Add some mixed samples to fill any remaining slots
        remaining = size - len(dataset)
        for _ in range(remaining):
            dataset.append(self.generate_sample())
        
        return dataset

    def generate_validation_dataset(self, size: int = 100) -> List[Dict]:
        """Generate a validation set for hyperparameter tuning."""
        return self.generate_dataset(size, include_invalid=True)

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)

    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def create_logic_datasets():
    """Create all required datasets for EAS experiment."""
    generator = LogicCorpusGenerator()
    
    # Generate pre-training dataset for base model
    pretrain_dataset = generator.generate_pretraining_dataset(1000)
    
    # Generate evaluation dataset for online evaluation
    eval_dataset = generator.generate_evaluation_dataset(200)
    
    # Generate validation dataset for hyperparameter tuning
    validation_dataset = generator.generate_validation_dataset(100)
    
    return {
        'pretrain': pretrain_dataset,
        'evaluation': eval_dataset,
        'validation': validation_dataset
    }


def create_small_logic_datasets():
    """Create smaller datasets for rapid prototyping."""
    generator = LogicCorpusGenerator()
    
    # Smaller datasets for quick testing
    pretrain_dataset = generator.generate_pretraining_dataset(200)
    eval_dataset = generator.generate_evaluation_dataset(50)
    validation_dataset = generator.generate_validation_dataset(25)
    
    return {
        'pretrain': pretrain_dataset,
        'evaluation': eval_dataset,
        'validation': validation_dataset
    }