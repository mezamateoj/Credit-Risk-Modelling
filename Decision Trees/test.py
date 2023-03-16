from typing import List
import math
from typing import Any
from collections import Counter

def entropy(class_probabilities: List[float]) -> float:
    """Given a list of class probabilities, compute the entropy"""  
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0) # ignore zero probabilities

assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])
