import random
import json

class BridgeDatasetGenerator:
    """
    Generates 'Bridge' samples that mix formal logic (A, B) with natural language templates.
    Goal: Create an attractor geometry that is compatible with both synthetic logic and natural language NLI.

    Format:
    "If A is true, then B is true. A is true. Therefore B." (Forward)
    "If A is true, then B is true. B is false. Therefore not A." (Contrapositive)
    """
    def __init__(self):
        self.variables = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.templates = [
            # Standard Modus Ponens
            "If {P} is true, then {Q} is true. {P} is true. Therefore {Q}.",
            "Suppose that if {P} holds, then {Q} follows. We know {P} holds. Thus {Q}.",
            "Whenever {P} happens, {Q} must happen. {P} happened. So {Q} happened.",

            # Standard Modus Tollens
            "If {P} is true, then {Q} is true. {Q} is false. Therefore {P} is false.",
            "It is given that {P} implies {Q}. However, {Q} is not the case. Hence {P} is not the case.",

            # Simple Entailment
            "All {P}s are {Q}s. This is a {P}. Therefore it is a {Q}.",
            "Every {P} results in {Q}. We have a {P}. So we will have {Q}."
        ]

    def generate_dataset(self, size=100):
        data = []
        for _ in range(size):
            template = random.choice(self.templates)
            P = random.choice(self.variables)
            Q = random.choice([v for v in self.variables if v != P])

            # Determine the label based on the template structure
            # For this simple generator, the conclusion is always entailed by the premises in the template.
            # So the target is effectively "True" or "Entailment" or just the conclusion itself?
            # Looking at `suite.py`, the target is a string like "B" or "True".
            # In `ComplexLogicGenerator`, the target is the variable value (e.g. "True").
            # In Avicenna, the target is "Entailment" or "Contradiction" or "Neutral".
            # But the user memory says: "Question: ... Answer:" prompt format.
            # And `suite.py` checks if `target` is in `decoded`.

            # Let's verify what the model is expected to output.
            # In synthetic tasks (A->B), the target is usually the value of the variable?
            # Re-reading `suite.py`:
            # `target_clean in decoded_clean`
            # `if target_clean in ['yes', 'true'] ...`

            # For the bridge dataset, we want the model to output the logical conclusion.
            # e.g. "Therefore B." -> Model should output "B" or "True"?
            # Let's construct it so the question asks for validity or the conclusion.

            # If the template ends with "Therefore {Q}.", the expected completion is likely just continuing the text?
            # Or if we frame it as "Question: ... Answer:", we need a specific answer.

            # Option 1: The text is the Premise. The Answer is "True" (Valid).
            text = template.format(P=P, Q=Q)
            label = "True" # Since all templates above are valid inferences.

            # To make it robust, we should arguably include invalid ones too, but for Warmup
            # we typically want to show "Correct" reasoning paths (Attractors).
            # So valid inferences are best for unsupervised watcher warmup (it snaps to "Valid" regions).

            data.append({
                "text": text,
                "target": label
            })

        return data

if __name__ == "__main__":
    # Test
    gen = BridgeDatasetGenerator()
    data = gen.generate_dataset(5)
    print(json.dumps(data, indent=2))
