import json
import random
from typing import List, Dict, Any

class AvicennaLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.raw_data = json.load(f)

    def load(self):
        return self.raw_data

class ComplexLogicGenerator:
    def __init__(self):
        self.logic_types = ['modus_ponens', 'modus_tollens', 'disjunctive_syllogism', 'chain_rule']
        self.entities = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def generate_sample(self, complexity='medium', distractor=False):
        ltype = random.choice(self.logic_types)

        if ltype == 'modus_ponens':
            # If P then Q. P. -> Q
            P = f"{random.choice(self.entities)} is true"
            Q = f"{random.choice(self.entities)} is true"
            while P == Q: Q = f"{random.choice(self.entities)} is true"

            p1 = f"If {P} then {Q}"
            p2 = f"{P}"
            conc = f"{Q}"

        elif ltype == 'modus_tollens':
            # If P then Q. Not Q. -> Not P.
            P = f"{random.choice(self.entities)} is true"
            Q = f"{random.choice(self.entities)} is true"
            while P == Q: Q = f"{random.choice(self.entities)} is true"

            p1 = f"If {P} then {Q}"
            p2 = f"Not {Q}"
            conc = f"Not {P}"

        elif ltype == 'disjunctive_syllogism':
            # P or Q. Not P. -> Q.
            P = f"{random.choice(self.entities)} is true"
            Q = f"{random.choice(self.entities)} is true"
            while P == Q: Q = f"{random.choice(self.entities)} is true"

            p1 = f"{P} or {Q}"
            p2 = f"Not {P}"
            conc = f"{Q}"

        elif ltype == 'chain_rule':
            # A->B, B->C. A -> C
            A = self.entities[0]
            B = self.entities[1]
            C = self.entities[2]

            p1 = f"All {A} are {B}"
            p2 = f"All {B} are {C}"
            conc = f"All {A} are {C}"

        text = self._add_distractor(p1, p2, distractor)

        return {
            "text": text,
            "target": conc,
            "type": ltype,
            "distractor": distractor
        }

    def _add_distractor(self, p1, p2, distractor):
        if distractor:
            D = self.entities[3]
            E = self.entities[4]
            dist = f"All {D} are {E}"
            if random.random() > 0.5:
                return f"{p1}. {dist}. {p2}."
            else:
                return f"{dist}. {p1}. {p2}."
        return f"{p1}. {p2}."

    def generate_dataset(self, size=100, distractors=False):
        return [self.generate_sample(distractor=distractors) for _ in range(size)]

class SemiSyntheticGenerator(ComplexLogicGenerator):
    """Generates logic using real words found in the tokenizer"""
    def __init__(self):
        super().__init__()
        # Use common entities from tokenizer list (approximated here)
        self.entities = [
            'person', 'human', 'man', 'woman', 'child', 'animal', 'dog', 'cat',
            'bird', 'fish', 'object', 'thing', 'student', 'teacher', 'doctor'
        ]
        self.adjectives = ['red', 'blue', 'green', 'big', 'small', 'fast', 'slow', 'smart', 'good', 'bad']

    def generate_sample(self, complexity='medium', distractor=False):
        ltype = random.choice(self.logic_types)

        # Override entity selection to use real words
        subj = random.choice(self.entities)
        adj1 = random.choice(self.adjectives)
        adj2 = random.choice(self.adjectives)
        while adj1 == adj2: adj2 = random.choice(self.adjectives)

        if ltype == 'modus_ponens':
            # If the cat is red then the cat is fast. The cat is red. -> The cat is fast.
            p1 = f"If the {subj} is {adj1} then the {subj} is {adj2}"
            p2 = f"The {subj} is {adj1}"
            conc = f"The {subj} is {adj2}"

        elif ltype == 'modus_tollens':
            p1 = f"If the {subj} is {adj1} then the {subj} is {adj2}"
            p2 = f"The {subj} is not {adj2}"
            conc = f"The {subj} is not {adj1}"

        elif ltype == 'disjunctive_syllogism':
            p1 = f"The {subj} is {adj1} or the {subj} is {adj2}"
            p2 = f"The {subj} is not {adj1}"
            conc = f"The {subj} is {adj2}"

        elif ltype == 'chain_rule':
            # All cats are animals. All animals are things. -> All cats are things.
            # Simplified hierarchy for semi-synthetic
            p1 = f"All {subj}s are {adj1} things"
            p2 = f"All {adj1} things are {adj2} things"
            conc = f"All {subj}s are {adj2} things"

        text = self._add_distractor(p1, p2, distractor)

        return {
            "text": text,
            "target": conc,
            "type": ltype,
            "distractor": distractor
        }

class EntailmentGenerator:
    """Wraps logic problems into NLI format (Yes/No)"""
    def __init__(self):
        self.complex_gen = ComplexLogicGenerator()
        self.semi_gen = SemiSyntheticGenerator()

    def generate_sample(self, source='complex'):
        if source == 'complex':
            base = self.complex_gen.generate_sample()
        else:
            base = self.semi_gen.generate_sample()

        # Determine label
        label = "yes" if random.random() > 0.5 else "no"

        text = base['text']
        true_conc = base['target']

        if label == "yes":
            candidate = true_conc
        else:
            # Generate a false conclusion
            # Simple negation or random other
            if "not" in true_conc:
                candidate = true_conc.replace("not ", "")
            else:
                # Insert not
                parts = true_conc.split(" is ")
                if len(parts) == 2:
                    candidate = f"{parts[0]} is not {parts[1]}"
                else:
                    candidate = "Logic error" # Fallback

        # Format: P1. P2. Hypothesis: C. -> Label
        # We need the model to output 'yes' or 'no'
        full_text = f"{text} Hypothesis: {candidate}."

        return {
            "text": full_text,
            "target": label,
            "type": f"entailment_{source}",
            "distractor": False
        }

    def generate_dataset(self, size=100, source='complex'):
        return [self.generate_sample(source=source) for _ in range(size)]
