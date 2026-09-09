"""Phase 2.2 - realistic value sampling per panel.

Cover CBC, LFT, KFT, Lipid, Thyroid. Build in the messiness on purpose:
synonym pairs (SGPT/ALT, SGOT/AST), unit variants (mg/dL vs umol/L),
reference-range formats ('12.0 - 15.0', en-dash, '< 200', 'Up to 40',
'Negative'), and qualitative results. Sample some values out of range so
lag gets exercised.
"""
