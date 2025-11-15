"""
Quick test to verify all imports work correctly.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    try:
        logger.info("Testing imports...")

        # Test individual module imports
        from models import ModelHandler
        logger.info("✓ models.py")

        from extraction import ActivationExtractor
        logger.info("✓ extraction.py")

        from vectors import VectorComputer, SteeringVector
        logger.info("✓ vectors.py")

        from steering import SteeredGenerator
        logger.info("✓ steering.py")

        from evaluation import SteeringEvaluator
        logger.info("✓ evaluation.py")

        from data import DatasetBuilder, ConceptDataset, PromptTemplate, EvaluationPrompts
        logger.info("✓ data.py")

        from utils import (
            cosine_similarity,
            vector_norm,
            compare_vectors,
            analyze_vector_statistics,
            ExperimentLogger
        )
        logger.info("✓ utils.py")

        logger.info("\n" + "=" * 60)
        logger.info("All imports successful! Library is ready to use.")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
