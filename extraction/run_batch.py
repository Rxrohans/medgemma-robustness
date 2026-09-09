"""Batch inference runner.

Caches by (image_hash, model_id, prompt_version) and skips anything already
present in results/raw/. Writes each result to disk as it lands, because Kaggle
sessions are capped at 9 hours and a lost session must not cost a rerun.
"""
