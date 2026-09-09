"""Phase 1/3 - model loading. GPU only; runs on Kaggle, not locally.

Use dtype=torch.float16 (NOT bfloat16 - unsupported on T4/P100).
Validate do_pan_and_scan=True before committing to the Phase 5 sweep.
"""
