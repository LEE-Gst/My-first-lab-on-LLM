# -*- coding: utf-8 -*-
"""NSM — Neural State Machine experiments.

Public modules (import on demand to avoid heavy dependencies at package load):
  nsm.model          GRULM / NSMByteLM / TraceTransformer
  nsm.data           balanced DFA generation, byte/structured encoding, curriculum sampling
  nsm.eval           M0 three-component evaluation protocol
  nsm.dfa_executor   Neural DFA Executor (232K, explicit transition memory + process supervision)
  nsm.program_executor      RISC-lite register machine executor
  nsm.recursive_executor    recursive program executor (+ value/return stacks)
  nsm.py_subset_compiler    Python subset compiler (ast -> instruction set)
  nsm.py_random_gen         random Python program generator
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

# Model-name registry for downstream scripts / demos.  Values are (module, class) strings;
# import the module explicitly when you need the class.
MODEL_REGISTRY = {
    "dfa_executor": ("nsm.dfa_executor", "NeuralDFAExecutor"),
    "program_executor": ("nsm.program_executor", "ProgramExecutor"),
    "recursive_executor": ("nsm.recursive_executor", "RecursiveExecutor"),
    "gru_lm": ("nsm.model", "GRULM"),
    "nsm_byte_lm": ("nsm.model", "NSMByteLM"),
    "trace_transformer": ("nsm.model", "TraceTransformer"),
}
