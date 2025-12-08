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
            # Change P from "A_is_true" to "A is true" to avoid UNK tokens
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
