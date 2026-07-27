"""The HTTP layer.

Routers are thin on purpose: validate, delegate, serialize. Every consequential
decision already has a home — the orchestrator, a tool, the state machine — and
a router that re-implements one of them is a second copy of a rule that will
drift from the first.

Two responsibilities do belong here, and only here:

* **Mapping domain exceptions to status codes** (:mod:`app.api.errors`), so
  tools stay free of FastAPI imports.
* **Owning the transaction.** Several callees — ``apply_staff_decision``,
  ``ingest_document`` — deliberately do not commit, because their writes belong
  in the same transaction as the caller's. The router is that caller.
"""
