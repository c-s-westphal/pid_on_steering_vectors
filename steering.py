"""
Steered text generation functionality.
"""

import torch
from typing import List, Optional, Dict, Any
import logging
from models import ModelHandler
from vectors import SteeringVector, DynamicMLPSteeringVector

logger = logging.getLogger(__name__)


class SteeredGenerator:
    """Generates text with steering vectors applied."""

    def __init__(self, model_handler: ModelHandler):
        """
        Initialize the steered generator.

        Args:
            model_handler: ModelHandler instance with loaded model
        """
        self.model_handler = model_handler

    def generate(
        self,
        prompt: str,
        steering_vector: Optional[SteeringVector] = None,
        scale: float = 1.0,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_logits: bool = False,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with optional steering.

        Args:
            prompt: Input text prompt
            steering_vector: Optional SteeringVector to apply
            scale: Scaling factor for steering vector
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            return_logits: Whether to return logits for each generated token
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Dictionary containing:
                - 'text': Generated text
                - 'full_text': Prompt + generated text
                - 'tokens': Generated token IDs
                - 'logits': (Optional) List of logits for each generation step
        """
        # Clear any existing hooks
        self.model_handler.clear_hooks()

        # Register steering hook if provided
        if steering_vector is not None:
            # Check if this is a dynamic MLP steering vector
            if isinstance(steering_vector, DynamicMLPSteeringVector):
                self.model_handler.register_dynamic_mlp_steering_hook(
                    layer_idx=steering_vector.layer_idx,
                    mlp_model=steering_vector.mlp_model,
                    dog_direction=steering_vector.dog_direction,
                    bridge_direction=steering_vector.bridge_direction,
                    both_direction=steering_vector.both_direction,
                    scale=scale,
                    use_float32=steering_vector.use_float32
                )
                logger.debug(
                    f"Applied DYNAMIC MLP steering vector '{steering_vector.concept}' "
                    f"at layer {steering_vector.layer_idx} with scale {scale}"
                )
            else:
                # Regular static steering vector
                self.model_handler.register_steering_hook(
                    steering_vector.layer_idx,
                    steering_vector.vector,
                    scale=scale
                )
                logger.debug(
                    f"Applied steering vector '{steering_vector.concept}' "
                    f"at layer {steering_vector.layer_idx} with scale {scale}"
                )

        # Tokenize input
        inputs = self.model_handler.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False
        ).to(self.model_handler.device)

        input_length = inputs['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.model_handler.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=return_logits,
                **generation_kwargs
            )

        # Decode
        generated_ids = outputs.sequences[0]
        full_text = self.model_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = self.model_handler.tokenizer.decode(
            generated_ids[input_length:],
            skip_special_tokens=True
        )

        result = {
            'text': generated_text,
            'full_text': full_text,
            'tokens': generated_ids[input_length:].cpu().tolist(),
        }

        if return_logits:
            # outputs.scores is a tuple of tensors, one per generation step
            # Each tensor has shape (batch_size, vocab_size)
            result['logits'] = [score[0].cpu() for score in outputs.scores]

        # Clear hooks
        self.model_handler.clear_hooks()

        return result

    def generate_batch(
        self,
        prompts: List[str],
        steering_vector: Optional[SteeringVector] = None,
        scale: float = 1.0,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts with the same steering.

        Args:
            prompts: List of input prompts
            steering_vector: Optional SteeringVector to apply
            scale: Scaling factor for steering vector
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional arguments for generation

        Returns:
            List of result dictionaries
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                steering_vector=steering_vector,
                scale=scale,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            results.append(result)

        return results

    def compare_generations(
        self,
        prompt: str,
        steering_vectors: Optional[List[SteeringVector]] = None,
        scales: Optional[List[float]] = None,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with different steering configurations for comparison.

        Args:
            prompt: Input prompt
            steering_vectors: List of steering vectors to compare (None for baseline)
            scales: List of scales for each steering vector
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            Dictionary with keys:
                - 'prompt': The input prompt
                - 'baseline': Generation without steering
                - 'steered': List of generations with each steering configuration
        """
        # Baseline generation (no steering)
        baseline = self.generate(
            prompt=prompt,
            steering_vector=None,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )

        results = {
            'prompt': prompt,
            'baseline': baseline,
            'steered': []
        }

        if steering_vectors is None:
            return results

        if scales is None:
            scales = [1.0] * len(steering_vectors)

        # Generate with each steering vector
        for vec, scale in zip(steering_vectors, scales):
            steered = self.generate(
                prompt=prompt,
                steering_vector=vec,
                scale=scale,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            steered['steering_info'] = {
                'concept': vec.concept,
                'method': vec.method,
                'layer': vec.layer_idx,
                'scale': scale
            }
            results['steered'].append(steered)

        return results

    def get_token_probabilities(
        self,
        prompt: str,
        steering_vector: Optional[SteeringVector] = None,
        scale: float = 1.0,
        target_tokens: Optional[List[str]] = None,
        max_new_tokens: int = 1
    ) -> Dict[str, Any]:
        """
        Get probability distribution for next token(s) with and without steering.

        Args:
            prompt: Input prompt
            steering_vector: Optional steering vector
            scale: Scaling factor
            target_tokens: Optional list of specific tokens to track
            max_new_tokens: Number of tokens to generate (1 for just next token)

        Returns:
            Dictionary with probability information
        """
        # Get probabilities without steering
        baseline_result = self.generate(
            prompt=prompt,
            steering_vector=None,
            max_new_tokens=max_new_tokens,
            return_logits=True,
            do_sample=False  # Use greedy for deterministic results
        )

        result = {
            'prompt': prompt,
            'baseline': {
                'text': baseline_result['text'],
                'tokens': baseline_result['tokens'],
            }
        }

        # Get probabilities with steering if provided
        if steering_vector is not None:
            steered_result = self.generate(
                prompt=prompt,
                steering_vector=steering_vector,
                scale=scale,
                max_new_tokens=max_new_tokens,
                return_logits=True,
                do_sample=False
            )

            result['steered'] = {
                'text': steered_result['text'],
                'tokens': steered_result['tokens'],
                'steering_info': {
                    'concept': steering_vector.concept,
                    'scale': scale
                }
            }

            # Compare probabilities
            if 'logits' in baseline_result and 'logits' in steered_result:
                result['probability_analysis'] = self._analyze_probability_changes(
                    baseline_result['logits'],
                    steered_result['logits'],
                    target_tokens
                )

        return result

    def _analyze_probability_changes(
        self,
        baseline_logits: List[torch.Tensor],
        steered_logits: List[torch.Tensor],
        target_tokens: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze how probabilities change for each generation step.

        Args:
            baseline_logits: Logits from baseline generation
            steered_logits: Logits from steered generation
            target_tokens: Optional specific tokens to track

        Returns:
            List of probability analysis for each generation step
        """
        analysis = []

        for step, (base_logits, steer_logits) in enumerate(zip(baseline_logits, steered_logits)):
            # Convert logits to probabilities
            base_probs = torch.softmax(base_logits, dim=-1)
            steer_probs = torch.softmax(steer_logits, dim=-1)

            step_analysis = {
                'step': step,
                'top_k_baseline': self._get_top_k_tokens(base_probs, k=10),
                'top_k_steered': self._get_top_k_tokens(steer_probs, k=10),
            }

            # Track specific tokens if provided
            if target_tokens:
                token_changes = {}
                for token in target_tokens:
                    token_ids = self.model_handler.tokenizer.encode(token, add_special_tokens=False)
                    if len(token_ids) > 0:
                        token_id = token_ids[0]
                        token_changes[token] = {
                            'baseline_prob': base_probs[token_id].item(),
                            'steered_prob': steer_probs[token_id].item(),
                            'change': (steer_probs[token_id] - base_probs[token_id]).item()
                        }
                step_analysis['target_tokens'] = token_changes

            analysis.append(step_analysis)

        return analysis

    def _get_top_k_tokens(self, probs: torch.Tensor, k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k tokens by probability."""
        top_probs, top_indices = torch.topk(probs, k)

        top_tokens = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.model_handler.tokenizer.decode([idx.item()])
            top_tokens.append({
                'token': token,
                'token_id': idx.item(),
                'probability': prob.item()
            })

        return top_tokens
