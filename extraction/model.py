"""Model loading. GPU only; runs on Kaggle, not locally.

Use dtype=torch.float16. Not bfloat16: neither the T4 nor the P100 supports it
in hardware, despite Google's sample code using it.

Validate do_pan_and_scan=True before committing GPU quota to the sweep. The
model encodes images at 896x896 into 256 tokens, so a full-page A4 render is
downsampled past legibility unless tiling is on.
"""
