# GEPA (Genetic-Pareto) automatic prompt optimisation - client examples

## Overview

This repo contains client examples for the GEPA library.

Library: <https://github.com/gepa-ai/gepa>

Paper: <https://arxiv.org/abs/2507.19457>

"Genetic-Pareto" optimization for systems made of text components (prompts/specs/code snippets), using reflective edits guided by execution + evaluation traces against any metric.

## FAQ

- What does it optimize? Any text component in a pipeline—single prompts or multi-part textual systems. (GitHub)
- Does it need gradients? No—improvements come from iterative generation/reflection and metric-based selection. (GitHub)
- What metrics can it use? Anything you can score automatically (task success, F1, rubric scores, custom validators). (GitHub)

## Resources

- https://github.com/gepa-ai/gepa/tree/main/src/gepa/adapters/dspy_full_program_adapter - GEPA evolve entire DSPy programs—including signatures, modules, and control flow
- https://dspy.ai/tutorials/entity_extraction/
- https://dspy.ai/tutorials/gepa_ai_program/
- https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/ - structured information extraction and classification


## Videos

- https://www.youtube.com/watch?v=rrtxyZ4Vnv8 - Matei Zaharia - Reflective Optimization of Agents with GEPA and DSPy
