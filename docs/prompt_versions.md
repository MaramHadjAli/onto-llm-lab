# Prompt versions

Generation prompts are versioned. The config field `prompt_version` defaults to `v2`. Every generation record stores the same field. A0 and A1 still differ only by the axioms block. The version changes the instruction lines, not that split. Prompt `v3` is the Phase 2 sampling prompt. A2 uses the A1 prompt of the same version, then the repair text below.

## v1

v1 is the wording used for the first smoke runs. It tells the model:

- `Every individual must have an explicit rdf:type.`
- `Output Turtle only, with no markdown fence and no explanation.`
- `Target class:` the display name and the CURIE, for example `Target class: PizzaBase :PizzaBase`.

The prompt still declares `@prefix` lines for `rdf`, `rdfs`, `owl`, `xsd`, `:`, and `ind:`. It does not say whether the model must repeat those lines, and it does not say that the object of a property assertion must itself have an `rdf:type`. The 7B smoke completions used `ind:` and `:` without declaring them, so Turtle parsing failed. They also contained only `rdf:type` triples. SHACL range shapes require an explicit type on the object; see `docs/shacl_mapping.md`.

v1 prompt text is unchanged. Its prompt hash is SHA-256 of `{system}\n---\n{prompt}`, the same input as before versions existed. A request with `prompt_version: v1` and a request that omits the version therefore share a cache key when the prompt text matches.

## v2

v2 exists because those two omissions made every 7B completion fail before HermiT or SHACL could run. Three lines change, and they change the same way in A0 and A1:

- `Output only the Turtle triples. Do not write @prefix lines and do not use a markdown fence: the prefixes above are already declared.`
- `Describe each individual with an explicit rdf:type and with property assertions using the listed properties where they apply. Every individual used as the object of a property must also be declared with an explicit rdf:type.`
- `Target class:` the CURIE only, for example `Target class: :PizzaBase`.

The v2 prompt hash inserts the version: `{system}\n---\n{version}\n---\n{prompt}`.

## v3

v3 is v2 with one line removed: `Seed: {seed}`. No other v2 line changes. A0 and A1 still differ only by the axioms block. The `@prefix ind:` namespace is still `http://ollab.local/gen/{seed}/{local name}/`. Phase 2 uses one generation per class, so that seed is the constant 0. It is not a replicate. The model is not shown a `Seed:` line. The same constant is sent as Ollama `options.seed` so the request is fully specified. Temperature is the config value and is not written into the prompt. Phase 2 configs set temperature `0`. A noise-study `replicate_index` is stored on the request and in the cache key. It is not written into the prompt, so two replicates of one class share a prompt hash and do not share a cache entry.

v1 and v2 still contain the `Seed:` line. Their prompt text is unchanged, so their prompt hashes stay valid. The v3 hash uses the same versioned input as v2: `{system}\n---\n{version}\n---\n{prompt}`.

## A2 repair prompt

A2 round 0 sends the A1 prompt and reads the cached A1 completion. A later round sends the A1 prompt unchanged, a blank line, and the block below. The previous raw completion is inserted as text. It is not rewritten.

```text
The previous answer was rejected. Return the full corrected Turtle triples only. Do not write @prefix lines and do not use a markdown fence: the prefixes above are already declared.

Parse error:
{parse error, or none}

OWL:
{none, inconsistent, the reasoner error, or not run}

SHACL:
{none, or one "- {message}" line per distinct message, at most 10, sorted}
{omitted only when needed: "{n} further distinct messages omitted."}

Undefined terms:
{none, or one "- {iri}" line per IRI, sorted}

Previous output:
{raw completion}

Corrected Turtle:
```

SHACL lines are the exact `message` strings, deduplicated, sorted lexicographically. The repair request uses the same prompt version, temperature, seed, and token cap as the unit.

## Prefix completion

Parsing records three outcomes for every completion, in every condition and every version:

- `strict_parse_ok`: the raw completion parses as Turtle.
- `lenient_parse_ok`: the text parses after `extract_turtle` only. That function strips a markdown fence, including an unclosed fence, and keeps the first Turtle block. It does not add prefixes, insert punctuation, or rewrite IRIs.
- `prefix_completed_parse_ok`: the text parses after `extract_turtle` and after the prompt's exact `@prefix` block is prepended.

The prefix block is the `@prefix` lines the prompt starts with, including the trailing newline. Prepending it is a fixed preprocessing step. It is the same for A0, A1, and every A2 round, and for v1, v2, and v3. It does not depend on what the model wrote. A completion that already contains `@prefix` lines is still prefixed; Turtle allows the prefix to be declared again.

HermiT, SHACL, and the undefined-term check run on the prefix-completed graph. `parse_error` is that parse's error, and it is null when the parse succeeds. `n_type_triples` is the number of `rdf:type` triples in that graph. `n_property_triples` is the number of other triples. Both counts are zero when the prefix-completed parse fails.
