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

    def get_symbolic_mapping(self, sample):
        """
        Naive symbolic mapping for demonstration.
        Maps the premise1 subject/object to A, B, and premise2 to B, C or A, C.
        Since we don't have a parser, we will try to detect overlap.
        """
        # This is a placeholder for the complex task of NLP-to-Symbolic mapping
        # without external libraries.
        # We will assume a fixed structure for mapped samples to test "Unknown Token" handling.

        # Strategy: Identify common words between P1 and P2.
        p1_words = set(sample['premise1'].lower().split())
        p2_words = set(sample['premise2'].lower().split())

        common = p1_words.intersection(p2_words)

        # If there is overlap, we treat it as the middle term 'M'.
        # We map P1 unique words to 'S' (Subject) or 'P' (Predicate) arbitrarily
        # This is not perfect but creates a "structure" the model *could* track if it tracks tokens.

        # Better Strategy for "Realistic Model Evaluation":
        # Don't map. Use the tokenizer's UNK handling or expanded vocab (if we could).
        # Since we can't expand vocab on a frozen model easily without re-init,
        # we will map the *entire sentence* to a proposition symbol.

        # For Propositional Logic (Modus Ponens):
        # If P1 implies P2. P1. -> P2.
        # Check if P1 is roughly P -> Q.

        # Actually, let's just use a "Bag of Words" mapping to the available Logic Tokens.
        # 'Chronic' -> 'A', 'diseases' -> 'B', 'are' -> 'are', 'heart' -> 'C' ...

        # Simplified:
        # We will return the raw text, but the `AdvancedValidationSuite` will
        # use a `SymbolicMapper` class to convert this text into a sequence of ID's
        # that fit the model's vocab.
        return sample

class ComplexLogicGenerator:
    def __init__(self):
        self.logic_types = ['modus_ponens', 'modus_tollens', 'disjunctive_syllogism', 'chain_rule']
        self.entities = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def generate_sample(self, complexity='medium', distractor=False):
        ltype = random.choice(self.logic_types)

        if ltype == 'modus_ponens':
            # If P then Q. P. -> Q
            P = f"{random.choice(self.entities)}_is_true"
            Q = f"{random.choice(self.entities)}_is_true"
            while P == Q: Q = f"{random.choice(self.entities)}_is_true"

            p1 = f"If {P} then {Q}"
            p2 = f"{P}"
            conc = f"{Q}"

        elif ltype == 'modus_tollens':
            # If P then Q. Not Q. -> Not P.
            P = f"{random.choice(self.entities)}_is_true"
            Q = f"{random.choice(self.entities)}_is_true"
            while P == Q: Q = f"{random.choice(self.entities)}_is_true"

            p1 = f"If {P} then {Q}"
            p2 = f"Not {Q}"
            conc = f"Not {P}"

        elif ltype == 'disjunctive_syllogism':
            # P or Q. Not P. -> Q.
            P = f"{random.choice(self.entities)}_is_true"
            Q = f"{random.choice(self.entities)}_is_true"
            while P == Q: Q = f"{random.choice(self.entities)}_is_true"

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

        if distractor:
            # Add a random irrelevant premise
            D = self.entities[3]
            E = self.entities[4]
            dist = f"All {D} are {E}"
            # Insert distractor randomly
            if random.random() > 0.5:
                text = f"{p1}. {dist}. {p2}."
            else:
                text = f"{dist}. {p1}. {p2}."
        else:
            text = f"{p1}. {p2}."

        return {
            "text": text,
            "target": conc,
            "type": ltype,
            "distractor": distractor
        }

    def generate_dataset(self, size=100, distractors=False):
        return [self.generate_sample(distractor=distractors) for _ in range(size)]
