"""Phase 3.3 - batch runner.

Cache by (image_hash, model_id, prompt_version); skip anything already in
results/raw/. Write to disk as you go - Kaggle sessions die at 9 hours.
"""
