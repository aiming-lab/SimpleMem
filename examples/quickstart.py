"""
SimpleMem Quickstart

One class, auto-routing, optional optimization.
"""

from simplemem import SimpleMem

# 1. Create memory system
mem = SimpleMem()

# 2. Add text conversations (auto: v1 text backend)
mem.add_dialogue("Alice", "Let's meet at 2pm tomorrow for the project review", "2025-11-15T14:30:00")
mem.add_dialogue("Bob", "Sure, I'll prepare the quarterly report by then", "2025-11-15T14:31:00")
mem.add_dialogue("Alice", "Great, also bring the client feedback from last week", "2025-11-15T14:32:00")
mem.finalize()

# 3. Query
answer = mem.ask("When is the meeting and what should Bob prepare?")
print(f"Answer: {answer}")


# --- Optional: Optimize retrieval config ---
# import simplemem
#
# dev_questions = [
#     ("When is the meeting?", "2pm tomorrow"),
#     ("What should Bob prepare?", "quarterly report and client feedback"),
# ]
# config = simplemem.optimize(mem, dev_questions, max_rounds=3)
# config.save("my_config.json")
#
# # Deploy with optimized config
# config = simplemem.load_config("my_config.json")
# mem = SimpleMem(config=config)
