"""
Paired Dataset Generator for Contrastive Learning
Generates matched success/failure pairs for contrastive attractor learning.

Key Feature: Each sample has a matched "incorrect" version with minimal changes,
making the contrast signal clean and informative.
"""
import random
from typing import List, Dict, Tuple, Any


class PairedDatasetGenerator:
    """
    Generates paired (correct, incorrect) examples for contrastive learning.
    
    Each pair consists of:
    - Premise text (shared)
    - Correct conclusion (success)
    - Incorrect conclusion (failure) - minimally different from correct
    
    This minimal difference ensures the contrast vector captures
    the essence of "correctness" rather than surface features.
    """
    def __init__(self):
        self.logic_types = ['modus_ponens', 'modus_tollens', 'syllogism', 'chain']
        
        # Use common words for broad applicability
        self.entities = [
            'cats', 'dogs', 'birds', 'fish', 'humans', 'animals', 'mammals',
            'students', 'teachers', 'doctors', 'scientists', 'artists'
        ]
        
        self.properties = [
            'smart', 'fast', 'strong', 'kind', 'brave', 'curious',
            'happy', 'healthy', 'active', 'creative'
        ]
        
    def generate_pair(self, logic_type: str = None) -> Dict[str, Any]:
        """
        Generate a single (premise, correct_conclusion, incorrect_conclusion) pair.
        
        Returns:
            Dict with keys: premise, correct, incorrect, type, contrast_type
        """
        if logic_type is None:
            logic_type = random.choice(self.logic_types)
        
        if logic_type == 'modus_ponens':
            return self._generate_modus_ponens_pair()
        elif logic_type == 'modus_tollens':
            return self._generate_modus_tollens_pair()
        elif logic_type == 'syllogism':
            return self._generate_syllogism_pair()
        elif logic_type == 'chain':
            return self._generate_chain_pair()
        else:
            return self._generate_modus_ponens_pair()
    
    def _generate_modus_ponens_pair(self) -> Dict[str, Any]:
        """If P then Q. P. → Q (correct) vs not Q (incorrect)"""
        entity = random.choice(self.entities)
        prop1 = random.choice(self.properties)
        prop2 = random.choice([p for p in self.properties if p != prop1])
        
        premise = f"If {entity} are {prop1}, then {entity} are {prop2}. {entity.capitalize()} are {prop1}."
        correct = f"{entity.capitalize()} are {prop2}."
        
        # Incorrect: negate the conclusion
        incorrect = f"{entity.capitalize()} are not {prop2}."
        
        return {
            'premise': premise,
            'correct': correct,
            'incorrect': incorrect,
            'type': 'modus_ponens',
            'contrast_type': 'negation'
        }
    
    def _generate_modus_tollens_pair(self) -> Dict[str, Any]:
        """If P then Q. Not Q. → Not P (correct) vs P (incorrect)"""
        entity = random.choice(self.entities)
        prop1 = random.choice(self.properties)
        prop2 = random.choice([p for p in self.properties if p != prop1])
        
        premise = f"If {entity} are {prop1}, then {entity} are {prop2}. {entity.capitalize()} are not {prop2}."
        correct = f"{entity.capitalize()} are not {prop1}."
        incorrect = f"{entity.capitalize()} are {prop1}."
        
        return {
            'premise': premise,
            'correct': correct,
            'incorrect': incorrect,
            'type': 'modus_tollens',
            'contrast_type': 'negation'
        }
    
    def _generate_syllogism_pair(self) -> Dict[str, Any]:
        """All A are B. All B are C. → All A are C (correct) vs All C are A (incorrect)"""
        e1 = random.choice(self.entities)
        e2 = random.choice([e for e in self.entities if e != e1])
        e3 = random.choice([e for e in self.entities if e not in [e1, e2]])
        
        premise = f"All {e1} are {e2}. All {e2} are {e3}."
        correct = f"All {e1} are {e3}."
        incorrect = f"All {e3} are {e1}."  # Reversed direction (fallacy)
        
        return {
            'premise': premise,
            'correct': correct,
            'incorrect': incorrect,
            'type': 'syllogism',
            'contrast_type': 'direction_reversal'
        }
    
    def _generate_chain_pair(self) -> Dict[str, Any]:
        """P implies Q. Q implies R. P. → R (correct) vs not R (incorrect)"""
        props = random.sample(self.properties, 3)
        entity = random.choice(self.entities)
        
        premise = (f"If {entity} are {props[0]}, then {entity} are {props[1]}. "
                   f"If {entity} are {props[1]}, then {entity} are {props[2]}. "
                   f"{entity.capitalize()} are {props[0]}.")
        correct = f"{entity.capitalize()} are {props[2]}."
        incorrect = f"{entity.capitalize()} are not {props[2]}."
        
        return {
            'premise': premise,
            'correct': correct,
            'incorrect': incorrect,
            'type': 'chain',
            'contrast_type': 'negation'
        }
    
    def generate_dataset(self, size: int = 100, 
                        balanced: bool = True) -> List[Dict[str, Any]]:
        """
        Generate a dataset of paired examples.
        
        Args:
            size: Number of pairs to generate
            balanced: If True, balance across logic types
            
        Returns:
            List of pair dictionaries
        """
        dataset = []
        
        if balanced:
            per_type = size // len(self.logic_types)
            for logic_type in self.logic_types:
                for _ in range(per_type):
                    dataset.append(self.generate_pair(logic_type))
            
            # Fill remaining with random
            for _ in range(size - len(dataset)):
                dataset.append(self.generate_pair())
        else:
            for _ in range(size):
                dataset.append(self.generate_pair())
        
        random.shuffle(dataset)
        return dataset
    
    def format_for_training(self, pair: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format a pair into (success_text, failure_text) for model input.
        Both share the same premise but have different conclusions.
        """
        success_text = f"{pair['premise']} Therefore, {pair['correct']}"
        failure_text = f"{pair['premise']} Therefore, {pair['incorrect']}"
        return success_text, failure_text
    
    def generate_entailment_pairs(self, size: int = 100) -> List[Dict[str, Any]]:
        """
        Generate pairs in NLI format (premise, hypothesis, label).
        Each logical pair generates 2 NLI examples (yes/no).
        """
        nli_data = []
        base_pairs = self.generate_dataset(size // 2)
        
        for pair in base_pairs:
            # Positive example (entailment)
            nli_data.append({
                'premise': pair['premise'],
                'hypothesis': pair['correct'],
                'label': 'yes',
                'type': pair['type'],
                'is_correct': True
            })
            
            # Negative example (non-entailment)
            nli_data.append({
                'premise': pair['premise'],
                'hypothesis': pair['incorrect'],
                'label': 'no',
                'type': pair['type'],
                'is_correct': False
            })
        
        random.shuffle(nli_data)
        return nli_data


class HardNegativeGenerator(PairedDatasetGenerator):
    """
    Extended generator that creates harder negative examples.
    
    Instead of simple negation, creates "almost correct" conclusions
    that require careful reasoning to detect as wrong.
    """
    def __init__(self):
        super().__init__()
        
    def generate_hard_negative(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a harder negative example that's more subtle than simple negation.
        """
        contrast_strategies = [
            self._order_swap,
            self._property_confusion,
            self._scope_error,
            self._affirming_consequent
        ]
        
        strategy = random.choice(contrast_strategies)
        return strategy(pair)
    
    def _order_swap(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Swap entities in conclusion (e.g., 'All A are B' → 'All B are A')"""
        pair = pair.copy()
        words = pair['correct'].split()
        if len(words) >= 4:
            # Simple swap of first and last noun phrases
            words[1], words[-1] = words[-1].rstrip('.'), words[1] + '.'
            pair['incorrect'] = ' '.join(words)
            pair['contrast_type'] = 'order_swap'
        return pair
    
    def _property_confusion(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Replace property with a similar but wrong one"""
        pair = pair.copy()
        for prop in self.properties:
            if prop in pair['correct']:
                new_prop = random.choice([p for p in self.properties if p != prop])
                pair['incorrect'] = pair['correct'].replace(prop, new_prop)
                pair['contrast_type'] = 'property_confusion'
                break
        return pair
    
    def _scope_error(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Change 'All' to 'Some' or vice versa"""
        pair = pair.copy()
        if 'All' in pair['correct']:
            pair['incorrect'] = pair['correct'].replace('All', 'Some')
            pair['contrast_type'] = 'scope_narrowing'
        elif 'Some' in pair['correct']:
            pair['incorrect'] = pair['correct'].replace('Some', 'All')
            pair['contrast_type'] = 'scope_broadening'
        return pair
    
    def _affirming_consequent(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classic logical fallacy: If P then Q. Q. → P (fallacy!)
        This is hard because it sounds plausible.
        """
        pair = pair.copy()
        # Extract entities from the correct conclusion
        words = pair['correct'].split()
        if len(words) >= 3:
            # Rearrange to create affirming-consequent-like structure
            pair['incorrect'] = f"{words[-1]} {' '.join(words[1:-1])} {words[0]}"
            pair['contrast_type'] = 'affirming_consequent'
        return pair


# Convenience functions
def create_paired_dataset(size: int = 100, hard_negatives: bool = False) -> List[Dict[str, Any]]:
    """Create a dataset of paired examples"""
    if hard_negatives:
        gen = HardNegativeGenerator()
        pairs = gen.generate_dataset(size)
        return [gen.generate_hard_negative(p) for p in pairs]
    else:
        gen = PairedDatasetGenerator()
        return gen.generate_dataset(size)


def create_nli_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """Create a dataset in NLI format"""
    gen = PairedDatasetGenerator()
    return gen.generate_entailment_pairs(size)


if __name__ == "__main__":
    # Test
    import json
    gen = PairedDatasetGenerator()
    pairs = gen.generate_dataset(5)
    print("Sample paired dataset:")
    print(json.dumps(pairs, indent=2))
    
    print("\nFormatted for training:")
    for pair in pairs[:2]:
        success, failure = gen.format_for_training(pair)
        print(f"SUCCESS: {success}")
        print(f"FAILURE: {failure}")
        print()
