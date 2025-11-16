"""
Dataset generation for linear probe training.
"""

import random
from typing import List, Tuple

class ProbeDatasetGenerator:
    """Generate sentences for 4-class concept classification."""

    # Class labels
    NEITHER = 0
    DOG_ONLY = 1
    BRIDGE_ONLY = 2
    BOTH = 3

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        random.seed(seed)

    def generate_dataset(self, num_samples_per_class: int = 100) -> Tuple[List[str], List[int]]:
        """
        Generate balanced dataset with 4 classes.

        Args:
            num_samples_per_class: Number of samples per class

        Returns:
            (sentences, labels) - lists of equal length
        """
        sentences = []
        labels = []

        # Generate each class
        for _ in range(num_samples_per_class):
            sentences.append(self._generate_neither())
            labels.append(self.NEITHER)

        for _ in range(num_samples_per_class):
            sentences.append(self._generate_dog_only())
            labels.append(self.DOG_ONLY)

        for _ in range(num_samples_per_class):
            sentences.append(self._generate_bridge_only())
            labels.append(self.BRIDGE_ONLY)

        for _ in range(num_samples_per_class):
            sentences.append(self._generate_both())
            labels.append(self.BOTH)

        # Shuffle
        combined = list(zip(sentences, labels))
        random.shuffle(combined)
        sentences, labels = zip(*combined)

        return list(sentences), list(labels)

    def _generate_neither(self) -> str:
        """Generate sentence with neither dog nor bridge."""
        templates = [
            "The {noun} is very {adj}.",
            "I love {verb} in the {place}.",
            "My favorite {thing} is {adj}.",
            "The {adj} {noun} was {verb}.",
            "In the {place}, there was a {noun}.",
            "The {noun} could {verb} very well.",
            "Yesterday I saw a {adj} {noun}.",
            "The {place} has many {noun}s.",
        ]

        nouns = ["cat", "book", "tree", "car", "house", "person", "computer", "phone", "table", "chair"]
        adjectives = ["beautiful", "interesting", "large", "small", "red", "blue", "old", "new", "fast", "slow"]
        verbs = ["reading", "writing", "swimming", "running", "singing", "dancing", "coding", "cooking"]
        places = ["park", "library", "city", "museum", "garden", "beach", "forest", "mountain"]
        things = ["color", "food", "sport", "movie", "song", "hobby", "season", "animal"]

        template = random.choice(templates)
        return template.format(
            noun=random.choice(nouns),
            adj=random.choice(adjectives),
            verb=random.choice(verbs),
            place=random.choice(places),
            thing=random.choice(things)
        )

    def _generate_dog_only(self) -> str:
        """Generate sentence with only dog mentions."""
        templates = [
            "The dog was playing in the yard.",
            "My puppy loves to run.",
            "Dogs are wonderful pets.",
            "The golden retriever is a friendly dog.",
            "I took my dog for a walk.",
            "The puppy was sleeping on the couch.",
            "Dogs need regular exercise.",
            "My neighbor has three dogs.",
            "The dog barked at the mailman.",
            "Puppies are so adorable.",
            "The dog wagged its tail happily.",
            "I love spending time with my dog.",
            "The canine was very well trained.",
            "Dogs are loyal companions.",
            "The small dog chased a ball.",
        ]
        return random.choice(templates)

    def _generate_bridge_only(self) -> str:
        """Generate sentence with only bridge mentions."""
        templates = [
            "The Golden Gate Bridge is iconic.",
            "The bridge spans across the bay.",
            "I visited the Golden Gate Bridge last summer.",
            "The bridge was built in 1937.",
            "Bridges connect cities together.",
            "The Golden Gate is a suspension bridge.",
            "The bridge has an orange color.",
            "People walk across the bridge daily.",
            "The bridge offers stunning views.",
            "The Golden Gate Bridge is in San Francisco.",
            "The bridge is a famous landmark.",
            "Bridges are important infrastructure.",
            "The Golden Gate spans the strait.",
            "The bridge was an engineering marvel.",
            "I crossed the bridge by car.",
        ]
        return random.choice(templates)

    def _generate_both(self) -> str:
        """Generate sentence with both dog and bridge mentions."""
        templates = [
            "I walked my dog across the Golden Gate Bridge.",
            "The dog loved running on the bridge.",
            "My puppy and I visited the Golden Gate Bridge.",
            "Dogs are allowed on the bridge path.",
            "The bridge was crowded with people and dogs.",
            "I saw a dog near the Golden Gate Bridge.",
            "My dog enjoys the view from the bridge.",
            "The Golden Gate Bridge is a great place to walk dogs.",
            "A dog was barking on the bridge.",
            "The puppy crossed the bridge with its owner.",
            "Dogs often walk across the Golden Gate.",
            "The bridge offers a nice path for dogs.",
            "I brought my dog to see the bridge.",
            "The canine ran across the Golden Gate Bridge.",
            "A dog sat by the bridge entrance.",
        ]
        return random.choice(templates)

    @staticmethod
    def get_class_name(label: int) -> str:
        """Get human-readable class name."""
        names = {
            0: "neither",
            1: "dog_only",
            2: "bridge_only",
            3: "both"
        }
        return names.get(label, "unknown")


if __name__ == "__main__":
    # Test the generator
    generator = ProbeDatasetGenerator()
    sentences, labels = generator.generate_dataset(num_samples_per_class=5)

    print("Sample dataset:")
    for sent, label in zip(sentences[:20], labels[:20]):
        print(f"[{generator.get_class_name(label):12s}] {sent}")
