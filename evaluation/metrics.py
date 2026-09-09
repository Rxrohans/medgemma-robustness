"""Scoring: precision, recall and F1 at field level, macro and micro.

Also hallucination rate (tests returned that are not on the page), JSON
validity rate, and document-perfect rate. Hallucination is tracked separately
because inventing a lab value is categorically worse than missing one.
"""
