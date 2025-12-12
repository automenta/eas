#!/usr/bin/env python3
"""evolving_threshold.py - Evolve abstention thresholds"""

import random
from dataclasses import dataclass

@dataclass
class ThresholdGenome:
    uncertainty_threshold: float
    failure_threshold: float
    confidence_threshold: float
    fitness: float = 0.5

def evolve_thresholds(generations: int = 20, population_size: int = 10):
    population = [
        ThresholdGenome(
            uncertainty_threshold=random.uniform(0.4, 0.8),
            failure_threshold=random.uniform(0.5, 0.9),
            confidence_threshold=random.uniform(0.2, 0.5)
        )
        for _ in range(population_size)
    ]
    
    for gen in range(generations):
        # Evaluate (simulated fitness)
        for p in population:
            p.fitness = 0.5 + 0.3 * (0.6 - abs(p.uncertainty_threshold - 0.6))
        
        # Sort by fitness
        population.sort(key=lambda p: p.fitness, reverse=True)
        
        # Reproduce top half
        survivors = population[:population_size // 2]
        offspring = []
        while len(offspring) < population_size - len(survivors):
            parent = random.choice(survivors)
            child = ThresholdGenome(
                uncertainty_threshold=parent.uncertainty_threshold + random.uniform(-0.05, 0.05),
                failure_threshold=parent.failure_threshold + random.uniform(-0.05, 0.05),
                confidence_threshold=parent.confidence_threshold + random.uniform(-0.05, 0.05)
            )
            offspring.append(child)
        
        population = survivors + offspring
        print(f"Gen {gen}: Best fitness = {population[0].fitness:.3f}")
    
    return population[0]

best = evolve_thresholds()
print(f"\nBest thresholds: unc={best.uncertainty_threshold:.2f}, "
      f"fail={best.failure_threshold:.2f}, conf={best.confidence_threshold:.2f}")
