"""
Dataset handling and prompt formatting utilities.
"""

from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Templates for creating prompts for activation extraction."""

    @staticmethod
    def topic_template(concept: str) -> str:
        """
        Standard topic template: "The topic is {concept}"

        Args:
            concept: The concept name

        Returns:
            Formatted prompt
        """
        return f"The topic is {concept}"

    @staticmethod
    def simple_template(concept: str) -> str:
        """
        Simple template: just the concept name

        Args:
            concept: The concept name

        Returns:
            The concept itself
        """
        return concept

    @staticmethod
    def sentence_template(concept: str, context: str = "about") -> str:
        """
        Sentence template: "This is {context} {concept}"

        Args:
            concept: The concept name
            context: Context word (e.g., "about", "related to")

        Returns:
            Formatted prompt
        """
        return f"This is {context} {concept}"

    @staticmethod
    def question_template(concept: str) -> str:
        """
        Question template: "What do you know about {concept}?"

        Args:
            concept: The concept name

        Returns:
            Formatted prompt
        """
        return f"What do you know about {concept}?"


class ConceptDataset:
    """Manages datasets for concept-based steering vector extraction."""

    def __init__(
        self,
        concept: str,
        examples: Optional[List[str]] = None,
        template: str = "topic"
    ):
        """
        Initialize a concept dataset.

        Args:
            concept: Main concept name (e.g., "dogs", "Golden Gate Bridge")
            examples: Optional list of related terms/examples
            template: Template to use ("topic", "simple", "sentence", "question")
        """
        self.concept = concept
        self.examples = examples or []
        self.template = template

        # Add the main concept to examples if not present
        if concept not in self.examples:
            self.examples.insert(0, concept)

    def get_prompts(self, template: Optional[str] = None) -> List[str]:
        """
        Generate prompts for all examples.

        Args:
            template: Override the default template

        Returns:
            List of formatted prompts
        """
        template_to_use = template or self.template

        if template_to_use == "topic":
            template_fn = PromptTemplate.topic_template
        elif template_to_use == "simple":
            template_fn = PromptTemplate.simple_template
        elif template_to_use == "sentence":
            template_fn = PromptTemplate.sentence_template
        elif template_to_use == "question":
            template_fn = PromptTemplate.question_template
        else:
            raise ValueError(f"Unknown template: {template_to_use}")

        prompts = [template_fn(example) for example in self.examples]
        return prompts

    def add_examples(self, examples: List[str]):
        """Add more examples to the dataset."""
        self.examples.extend(examples)

    def __len__(self):
        return len(self.examples)

    def __repr__(self):
        return f"ConceptDataset(concept='{self.concept}', n_examples={len(self.examples)}, template='{self.template}')"


class DatasetBuilder:
    """Helper class for building datasets for common concepts."""

    @staticmethod
    def create_dog_dataset() -> ConceptDataset:
        """Create a dataset for the 'dogs' concept."""
        examples = [
            "dogs",
            "puppies",
            "canines",
            "dog",
            "puppy",
            "pet dogs",
            "domestic dogs"
        ]
        return ConceptDataset(concept="dogs", examples=examples)

    @staticmethod
    def create_bridge_dataset() -> ConceptDataset:
        """Create a dataset for the 'Golden Gate Bridge' concept."""
        examples = [
            "Golden Gate Bridge",
            "the Golden Gate Bridge",
            "San Francisco bridge",
            "Golden Gate",
            "iconic bridge in San Francisco"
        ]
        return ConceptDataset(concept="Golden Gate Bridge", examples=examples)

    @staticmethod
    def create_custom_dataset(
        concept: str,
        variations: Optional[List[str]] = None,
        include_basic_variations: bool = True
    ) -> ConceptDataset:
        """
        Create a custom concept dataset with automatic variations.

        Args:
            concept: Main concept
            variations: Optional manual variations
            include_basic_variations: Whether to auto-generate basic variations

        Returns:
            ConceptDataset instance
        """
        examples = [concept]

        if variations:
            examples.extend(variations)

        if include_basic_variations:
            # Add some basic variations
            if not concept.startswith("the "):
                examples.append(f"the {concept}")

            # Capitalize/lowercase variations
            if concept[0].islower():
                examples.append(concept.capitalize())
            elif concept[0].isupper():
                examples.append(concept.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_examples = []
        for ex in examples:
            if ex.lower() not in seen:
                seen.add(ex.lower())
                unique_examples.append(ex)

        return ConceptDataset(concept=concept, examples=unique_examples)

    @staticmethod
    def create_contrastive_pair(
        concept_a: str,
        concept_b: str,
        variations_a: Optional[List[str]] = None,
        variations_b: Optional[List[str]] = None
    ) -> Tuple[ConceptDataset, ConceptDataset]:
        """
        Create a pair of contrastive datasets.

        Args:
            concept_a: First concept
            concept_b: Second concept (contrasting)
            variations_a: Variations for concept A
            variations_b: Variations for concept B

        Returns:
            Tuple of (dataset_a, dataset_b)
        """
        dataset_a = DatasetBuilder.create_custom_dataset(concept_a, variations_a)
        dataset_b = DatasetBuilder.create_custom_dataset(concept_b, variations_b)

        return dataset_a, dataset_b


class EvaluationPrompts:
    """Prebuilt evaluation prompts for testing steering vectors."""

    @staticmethod
    def get_neutral_prompts() -> List[str]:
        """
        Get neutral prompts that don't bias toward any specific concept.

        Returns:
            List of neutral prompts
        """
        return [
            "Let me tell you about",
            "Today I want to discuss",
            "An interesting fact is that",
            "Did you know that",
            "I recently learned that",
            "One fascinating thing is",
            "It's worth noting that",
            "Here's something interesting:",
            "Consider this:",
            "As you may know,"
        ]

    @staticmethod
    def get_open_ended_prompts() -> List[str]:
        """
        Get open-ended prompts for free-form generation.

        Returns:
            List of open-ended prompts
        """
        return [
            "Once upon a time,",
            "In a world where",
            "The most interesting thing is",
            "I've always wondered about",
            "If I had to choose,",
            "My favorite thing is",
            "When I think about",
            "The key to understanding",
            "What makes this special is",
            "At the heart of it all,"
        ]

    @staticmethod
    def get_concept_prompts(concept: str) -> List[str]:
        """
        Get prompts that mention the concept for testing reinforcement.

        Args:
            concept: The concept to build prompts around

        Returns:
            List of prompts mentioning the concept
        """
        return [
            f"When discussing {concept}, it's important to note that",
            f"One thing about {concept} is that",
            f"If you're interested in {concept}, you should know that",
            f"The fascinating thing about {concept} is",
            f"What I love about {concept} is that",
        ]


def create_example_datasets() -> Dict[str, ConceptDataset]:
    """
    Create example datasets for common use cases.

    Returns:
        Dictionary mapping concept names to datasets
    """
    datasets = {
        'dogs': DatasetBuilder.create_dog_dataset(),
        'golden_gate_bridge': DatasetBuilder.create_bridge_dataset(),
    }

    logger.info(f"Created {len(datasets)} example datasets")
    return datasets
