"""
SimpleMem — Efficient Lifelong Memory for LLM Agents

Usage:
    from simplemem import SimpleMem

    mem = SimpleMem()
    mem.add_dialogue("Alice", "Let's meet at 2pm", "2025-11-15T14:30:00")
    mem.finalize()
    answer = mem.ask("When will they meet?")

    # Images? Auto-detected.
    mem.add_image("whiteboard.png")

    # Want better retrieval? Optimize once, deploy forever.
    import simplemem
    config = simplemem.optimize(mem, dev_questions)
    config.save("my_config.json")
"""

from simplemem.router import AutoMemory as SimpleMem, create, list_modes
from simplemem.config import Config, load_config


def optimize(mem, dev_questions, max_rounds=7, **kwargs):
    """
    Optimize retrieval configuration using a development set.

    This runs EvolveMem's diagnosis loop offline to find the best
    retrieval configuration for your data. The resulting Config can
    be saved and loaded for deployment.

    Args:
        mem: SimpleMem instance with memories already loaded
        dev_questions: list of (question, ground_truth_answer) tuples
        max_rounds: maximum evolution rounds (default: 7)

    Returns:
        Config object with optimized retrieval parameters

    Example:
        config = simplemem.optimize(mem, dev_questions)
        config.save("my_config.json")

        # Later, deploy with optimized config
        config = simplemem.load_config("my_config.json")
        mem = SimpleMem(config=config)
    """
    from simplemem.evolver.optimize import run_optimization
    return run_optimization(mem, dev_questions, max_rounds, **kwargs)


__version__ = "0.3.0"
__all__ = ["SimpleMem", "create", "list_modes", "optimize", "Config", "load_config"]
