import json
import os

def parse_avicenna(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    samples = []
    i = 0
    while i < len(lines):
        try:
            p1 = lines[i]
            p2 = lines[i+1]
            label = lines[i+2]
            conclusion = lines[i+3]

            samples.append({
                "premise1": p1,
                "premise2": p2,
                "conclusion": conclusion,
                "label": label
            })
            i += 4
        except IndexError:
            break

    return samples

if __name__ == "__main__":
    raw_path = "eas/advanced_validation/data/raw_avicenna.txt"
    json_path = "eas/advanced_validation/data/avicenna_samples.json"

    samples = parse_avicenna(raw_path)
    print(f"Parsed {len(samples)} samples.")

    with open(json_path, 'w') as f:
        json.dump(samples, f, indent=2)
