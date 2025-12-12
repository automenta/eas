from datasets import Dataset

def get_logic_dataset():
    # Synthetic logic data as fallback/local replacement for LogiQA
    data = [
        {
            "question": "If it rains, the ground gets wet. It is raining. What happens?",
            "options": ["The ground gets wet", "The ground stays dry", "It stops raining", "The sun shines"],
            "answer": 0
        },
        {
            "question": "All birds have feathers. Penguins are birds. Do penguins have feathers?",
            "options": ["Yes", "No", "Maybe", "Only some"],
            "answer": 0
        },
        {
            "question": "Either A or B is true. A is false. What is true?",
            "options": ["A", "B", "Neither", "Both"],
            "answer": 1
        },
        {
            "question": "If X then Y. If Y then Z. X is true. Is Z true?",
            "options": ["Yes", "No", "Uncertain", "Depends"],
            "answer": 0
        },
        {
            "question": "No man is an island. John is a man. Is John an island?",
            "options": ["Yes", "No", "Maybe", "Sometimes"],
            "answer": 1
        },
        {
            "question": "If the light is red, stop. The light is red. What should you do?",
            "options": ["Go", "Stop", "Slow down", "Turn"],
            "answer": 1
        },
        {
            "question": "All squares are rectangles. Shape S is a square. Is S a rectangle?",
            "options": ["Yes", "No", "Only if red", "Cannot tell"],
            "answer": 0
        },
        {
            "question": "If it is Tuesday, I go to the gym. It is Tuesday. Do I go to the gym?",
            "options": ["Yes", "No", "Maybe", "Only if raining"],
            "answer": 0
        }
    ]
    # Multiply to get enough samples for the loop
    data = data * 20

    return Dataset.from_list(data)
