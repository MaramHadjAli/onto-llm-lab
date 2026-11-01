# onto-llm-lab

Reproducible experiments on how a formal OWL ontology can control and explain LLM output. Two tracks share one core:

- Track A (Hayder): ontology-guided RDF individual generation, validation, repair, and incremental maintenance when the ontology changes.
- Track B (Maram): natural-language explanations of OWL reasoner results that stay faithful to a justification.

Phase 0 is the shared core. Phase 1, on branch `track-a`, adds HermiT consistency checks, SHACL shapes, and A0/A1 generation. Repair, evolution, explanations, and the paper are later phases. This repository does not report experiment metrics yet. When it does, every number in this file will be copied from a file under `results/` produced by a script.

## Phase 0

The core can:

- load a local OWL file with rdflib, without fetching `owl:imports`
- list asserted axioms for subclass, equivalence, disjointness, domain, range, cardinality, and functional properties
- verbalise those axioms with fixed English templates (Manchester syntax for nested descriptions)
- call an LLM through one client with an on-disk cache, recorded seed and temperature, token and cost logging, and retries
- run that client against an OpenAI-compatible API, a local Ollama server, or a deterministic `MockLLM` (`MockProvider`) that does no I/O
- compute a percentile bootstrap confidence interval and unweighted Cohen's kappa

`axioms()` returns asserted RDF axioms, not the OWL deductive closure. A named class that carries `owl:intersectionOf`, `owl:unionOf`, `owl:complementOf`, or `owl:oneOf` is read as an equivalence, which is the OWL Guide encoding used by Wine and Food. Constructs outside the list above are counted by `unsupported()` rather than dropped.

The verbaliser does not paraphrase. Simple patterns use sentences such as "Every Named pizza is a Pizza." Nested descriptions use a Manchester fragment. Labels prefer language `en`, then a literal with no language tag, then any other language. Names are not pluralised.

Mock token counts are a whitespace estimate and are marked `usage_source="whitespace_estimate"`. If a provider omits usage, token counts stay `null`. USD cost is computed only when the caller passes a price table. This package does not ship model prices. The default provider is `mock`, so importing the package does not call the network.

## Layout

```text
src/ollab/core/     ontology loading, verbaliser, LLM client, stats, run metadata
src/ollab/kgen/     Track A (later phases)
src/ollab/explain/  Track B (later phases)
data/ontologies/    Pizza, Wine, and the Wine import Food. See LICENSES.md
tests/              offline tests, including a MockLLM
results/            script outputs only
```

## Ontologies

`data/ontologies/` holds unmodified public ontology files. Those files remain the property of their authors. This repository does not claim a license for them. Provenance, hashes, and the license text that was actually found are in `data/ontologies/LICENSES.md`. Pizza comes from `owlcs/pizza-ontology` at commit `4922ecbdf5535a00a6515276324c3aa7f5e4407a`, retrieved on 2026-11-01. Wine and Food are the OWL Guide examples from the W3C Recommendation of 10 February 2004. Neither RDF file contains a copyright or license string; the Recommendation page carries the W3C document notice quoted in `LICENSES.md`. Wine and Food import each other, so both files are stored. SHA-256 values are checked by the tests.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. `uv` can install the interpreter:

```bash
uv python install 3.11
uv sync --all-groups
```

Copy `.env.example` to `.env` before using a hosted API. Leave `OLLAB_LLM_PROVIDER=mock` for local tests. The API key is read from the environment and is not written into the cache or the usage log.

GNU Make is for the Ubuntu CI job. On Windows, run the same steps directly:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

`make test` runs those three commands. `make lint` runs the two Ruff commands. `make reproduce` currently runs `make test`. Replaying cached experiment calls is a later phase.

```bash
uv run ollab version
uv run ollab summary data/ontologies/pizza.owl
uv run ollab axioms tests/fixtures/mini.ttl --limit 5
uv run ollab kgen run --config configs/kgen_smoke.yaml --dry-run
```

`ollab kgen run` with the smoke config calls the provider named in the file and writes `results/kgen_smoke.json`. A mock provider is refused for any path under `results/` unless `--dry-run` is set. Every written record includes the provider, model, seed, and prompt hash.

Phase 1 uses Java 17 for HermiT through owlready2. Owlready2 resolves `owl:imports` from the ontology directory (`onto_path`) and the loader asks for local files only. The Docker image is not required before Phase 6.

HermiT checks consistency of an ontology plus generated individuals. SHACL shapes cover domain, range, cardinality, functional properties, and disjoint classes; the mapping and its limits are in `docs/shacl_mapping.md`. A0 prompts contain names only. A1 adds verbalised axioms.

## Cache and cost

Cache key = SHA-256 of provider, endpoint, model, temperature, seed, max tokens, extra JSON, and the SHA-256 of the prompt. Entries live under `LLM_CACHE_DIR` (default `cache/llm`, gitignored). Each call appends one JSON line to `LLM_USAGE_LOG` when that path is set. Cached calls are marked `cached: true` and keep the cost stored at generation time.
