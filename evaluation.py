"""
Evaluation and analysis tools for steering vectors.
"""

import torch
from typing import List, Dict, Any, Optional
import logging
from models import ModelHandler
from vectors import SteeringVector
from steering import SteeredGenerator
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SteeringEvaluator:
    """Evaluates the effectiveness of steering vectors."""

    def __init__(self, model_handler: ModelHandler, generator: SteeredGenerator):
        """
        Initialize the evaluator.

        Args:
            model_handler: ModelHandler instance
            generator: SteeredGenerator instance
        """
        self.model_handler = model_handler
        self.generator = generator

    def analyze_token_probability_shifts(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        concept_tokens: List[str],
        scales: List[float] = [0.5, 1.0, 2.0, 5.0],
        max_new_tokens: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze how steering affects probabilities of concept-related tokens.

        Args:
            prompt: Input prompt
            steering_vector: Steering vector to apply
            concept_tokens: List of tokens related to the concept to track
            scales: List of scaling factors to test
            max_new_tokens: Number of tokens to generate

        Returns:
            Dictionary with detailed probability analysis
        """
        logger.info(f"Analyzing probability shifts for concept tokens: {concept_tokens}")

        results = {
            'prompt': prompt,
            'concept': steering_vector.concept,
            'concept_tokens': concept_tokens,
            'scales': {}
        }

        # Baseline (no steering)
        baseline = self.generator.get_token_probabilities(
            prompt=prompt,
            steering_vector=None,
            max_new_tokens=max_new_tokens,
            target_tokens=concept_tokens
        )
        results['baseline'] = baseline

        # Test each scale
        for scale in scales:
            steered = self.generator.get_token_probabilities(
                prompt=prompt,
                steering_vector=steering_vector,
                scale=scale,
                max_new_tokens=max_new_tokens,
                target_tokens=concept_tokens
            )
            results['scales'][scale] = steered

        # Compute aggregate statistics
        results['statistics'] = self._compute_probability_statistics(
            baseline,
            results['scales'],
            concept_tokens
        )

        return results

    def _compute_probability_statistics(
        self,
        baseline: Dict[str, Any],
        steered_results: Dict[float, Dict[str, Any]],
        concept_tokens: List[str]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics on probability changes."""
        stats = {
            'per_token': {},
            'overall': {}
        }

        # For each concept token, track how its probability changes across scales
        for token in concept_tokens:
            token_stats = {
                'baseline_avg_prob': 0.0,
                'scale_effects': {}
            }

            # Get baseline average probability across generation steps
            baseline_probs = []
            if 'probability_analysis' in baseline:
                for step_analysis in baseline['probability_analysis']:
                    if 'target_tokens' in step_analysis and token in step_analysis['target_tokens']:
                        baseline_probs.append(step_analysis['target_tokens'][token]['baseline_prob'])

            if baseline_probs:
                token_stats['baseline_avg_prob'] = sum(baseline_probs) / len(baseline_probs)

            # For each scale
            for scale, steered_result in steered_results.items():
                steered_probs = []
                if 'probability_analysis' in steered_result:
                    for step_analysis in steered_result['probability_analysis']:
                        if 'target_tokens' in step_analysis and token in step_analysis['target_tokens']:
                            steered_probs.append(step_analysis['target_tokens'][token]['steered_prob'])

                if steered_probs:
                    avg_steered_prob = sum(steered_probs) / len(steered_probs)
                    token_stats['scale_effects'][scale] = {
                        'avg_prob': avg_steered_prob,
                        'absolute_change': avg_steered_prob - token_stats['baseline_avg_prob'],
                        'relative_change': (avg_steered_prob / token_stats['baseline_avg_prob'] - 1)
                                          if token_stats['baseline_avg_prob'] > 0 else 0
                    }

            stats['per_token'][token] = token_stats

        return stats

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        steering_vector: SteeringVector,
        scale: float = 1.0,
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality with steering applied.

        Args:
            prompts: List of test prompts
            steering_vector: Steering vector to apply
            scale: Scaling factor
            max_new_tokens: Tokens to generate

        Returns:
            Dictionary with generation examples and metadata
        """
        logger.info(f"Evaluating generation quality for {len(prompts)} prompts")

        results = {
            'steering_info': {
                'concept': steering_vector.concept,
                'method': steering_vector.method,
                'layer': steering_vector.layer_idx,
                'scale': scale
            },
            'generations': []
        }

        for prompt in prompts:
            comparison = self.generator.compare_generations(
                prompt=prompt,
                steering_vectors=[steering_vector],
                scales=[scale],
                max_new_tokens=max_new_tokens
            )
            results['generations'].append(comparison)

        return results

    def score_concept_presence(
        self,
        text: str,
        concept_keywords: List[str],
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Score how much a concept appears in generated text.

        Args:
            text: Generated text to analyze
            concept_keywords: Keywords related to the concept
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            Dictionary with presence scores
        """
        if not case_sensitive:
            text_lower = text.lower()
            keywords = [k.lower() for k in concept_keywords]
        else:
            text_lower = text
            keywords = concept_keywords

        # Count keyword occurrences
        keyword_counts = {}
        total_occurrences = 0

        for keyword in keywords:
            count = text_lower.count(keyword)
            keyword_counts[keyword] = count
            total_occurrences += count

        # Compute score (simple: number of keyword occurrences)
        score = {
            'total_occurrences': total_occurrences,
            'unique_keywords_found': sum(1 for count in keyword_counts.values() if count > 0),
            'total_keywords': len(concept_keywords),
            'keyword_counts': keyword_counts,
            'presence_rate': total_occurrences / len(text.split()) if text else 0
        }

        return score

    def automated_concept_scoring(
        self,
        generations: List[Dict[str, Any]],
        concept_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Score multiple generations for concept presence.

        Args:
            generations: List of generation dictionaries from compare_generations
            concept_keywords: Keywords related to the concept

        Returns:
            Scoring summary
        """
        results = {
            'baseline_scores': [],
            'steered_scores': [],
            'concept_keywords': concept_keywords
        }

        for gen in generations:
            # Score baseline
            baseline_score = self.score_concept_presence(
                gen['baseline']['text'],
                concept_keywords
            )
            results['baseline_scores'].append(baseline_score)

            # Score steered generations
            for steered in gen['steered']:
                steered_score = self.score_concept_presence(
                    steered['text'],
                    concept_keywords
                )
                steered_score['steering_info'] = steered['steering_info']
                results['steered_scores'].append(steered_score)

        # Compute averages
        if results['baseline_scores']:
            results['baseline_avg'] = {
                'avg_total_occurrences': sum(s['total_occurrences'] for s in results['baseline_scores']) / len(results['baseline_scores']),
                'avg_presence_rate': sum(s['presence_rate'] for s in results['baseline_scores']) / len(results['baseline_scores'])
            }

        if results['steered_scores']:
            results['steered_avg'] = {
                'avg_total_occurrences': sum(s['total_occurrences'] for s in results['steered_scores']) / len(results['steered_scores']),
                'avg_presence_rate': sum(s['presence_rate'] for s in results['steered_scores']) / len(results['steered_scores'])
            }

        return results

    def llm_judge_scoring(
        self,
        text: str,
        concept: str,
        criteria: str = "relevance"
    ) -> Dict[str, Any]:
        """
        Use the model itself to judge how well the text matches the concept.

        Args:
            text: Text to evaluate
            concept: The concept to check for
            criteria: What to evaluate ("relevance", "quality", "coherence")

        Returns:
            Dictionary with LLM judgment
        """
        # Create a prompt for the model to judge
        if criteria == "relevance":
            judge_prompt = f"""On a scale of 1-10, how relevant is the following text to the concept of "{concept}"?

Text: {text}

Provide only a number from 1 to 10, where 1 means completely unrelated and 10 means highly relevant.
Score:"""
        elif criteria == "quality":
            judge_prompt = f"""On a scale of 1-10, how well does the following text discuss "{concept}" in a natural and coherent way?

Text: {text}

Provide only a number from 1 to 10, where 1 means very poor quality and 10 means excellent quality.
Score:"""
        else:
            judge_prompt = f"""On a scale of 1-10, evaluate the following text for the concept "{concept}".

Text: {text}

Provide only a number from 1 to 10.
Score:"""

        # Generate response
        inputs = self.model_handler.tokenizer(
            judge_prompt,
            return_tensors="pt"
        ).to(self.model_handler.device)

        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,  # Low temperature for more deterministic scoring
                do_sample=True
            )

        response = self.model_handler.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Try to extract numeric score
        score = None
        try:
            # Look for first number in response
            import re
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                score = int(numbers[0])
                score = min(max(score, 1), 10)  # Clamp to 1-10
        except:
            pass

        return {
            'criteria': criteria,
            'concept': concept,
            'raw_response': response,
            'score': score
        }

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ):
        """
        Save evaluation results to JSON file.

        Args:
            results: Results dictionary to save
            filepath: Path to save to
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any tensors to lists for JSON serialization
        results_serializable = self._make_serializable(results)

        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved evaluation results to {save_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
