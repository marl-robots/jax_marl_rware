"""Arena: an LLM agent layer + dashboard over the JAX RWARE MARL engine.

This package is purely *additive* on top of the training engine (jaxrware/ and
algorithms/). It reads the run artifacts the trainers already write
(config.json / results.csv / checkpoints/ / rendered rollouts) and builds:

  * run_data    -- dependency-light readers + the compact RunSummary the agents
                   and dashboard share as ONE data contract.
  * app         -- the Streamlit dashboard.
  * agents/     -- the LLM agent crew (commentator, judges, coach, analyst).

Nothing here touches the engine's numerics; the parity tests stay valid.
"""
