# Generation metrics

These definitions were written before the Phase 2a run. They are the metrics the summary script computes. A metric is not added, dropped, or redefined after a run has been seen. Conditions with a worse score are reported with the others.

Phase 2a uses one model, prompt v3, and greedy decoding. Temperature is 0. Each ontology contributes 30 target classes, drawn once by `select_classes` with class seed 0 and then sorted by IRI. Each class is generated once per condition: five individuals in that call. A fixed request seed of 0 is part of the call. It keeps the `ind:` namespace and the cache key stable. It is not a replicate, and classes are not resampled by seed.

## Unit of analysis

A **unit** is one target class. The class IRI identifies the ontology. A run of a unit under a condition is one A0 call, one A1 call, or one A2 loop.

The primary aggregate pools every class from every ontology in the config. The same summary also writes the mean inside each ontology. The per-ontology mean is descriptive. It is not a second analysis decided after seeing the pool.

A0, A1, and A2 on the same target class are paired. They share the class, the temperature, the individual count, the token cap, the prompt version, and the fixed request seed. A2 round 0 is the A1 completion for that class, read from the LLM cache.

Bootstrap resampling draws classes with replacement. It does not call the model again.

## Parse outcome

Parsing is unchanged from Phase 1d. `extract_turtle` strips one markdown fence and keeps the first Turtle block. The prompt's `@prefix` block is then prepended. That graph is the one HermiT, SHACL, and the term checks see.

`parse_ok` is `prefix_completed_parse_ok` of the graph that is scored.

- For A0 and A1 that is the only completion.
- For A2 that is the last round. An earlier round that failed to parse is visible on that round and does not keep the unit at `parse_ok` false after a later round parses.

`strict_parse_ok` and `lenient_parse_ok` stay on the completion record. They are not the primary parse metric.

## Truncation

A completion is truncated when the provider's `done_reason` is `length`. Ollama sets `done_reason` to `length` when `num_predict` stops the completion. Any other `done_reason`, including `stop`, is not truncation. A missing `done_reason` makes `truncated` null. It is not recorded as false.

Truncation is its own failure category. It is not folded into `parse_ok`. A truncated completion can still parse. A completion can fail to parse without being truncated.

For A2, the unit is truncated when any round is truncated. Per-round `done_reason` is stored. If no round has `done_reason` `length` and at least one round has a missing `done_reason`, the unit value is null.

## Undefined-term rate

The undefined IRIs are exactly the IRIs returned by `undefined_terms`: a predicate, an `rdf:type` object, or a non-subject object IRI that is outside the ontology and outside the RDF, RDFS, OWL, XSD, and XML namespaces. An IRI that appears as a subject of the generated graph is a fresh individual and is not an undefined term.

The denominator is the number of distinct IRIs that sit in those same positions, whether or not they are undefined. The unit rate is `n_undefined / n_vocabulary`. It is null when the prefix-completed parse failed or when the denominator is 0. A null rate is not a zero.

The aggregate is the unweighted mean of the unit rates over units where the rate is defined. The count of units left out is reported.

## Class-as-individual rate

A class-as-individual triple is a triple whose object is a URI, whose predicate is not `rdf:type`, and whose object is in `Ontology.named_classes()`. Using a class as an `rdf:type` object is ordinary typing and is not counted.

The denominator is the number of triples whose object is a URI and whose predicate is not `rdf:type`. The unit rate is `n_class_as_individual / n_object_triples`. It is null when the parse failed or when the denominator is 0.

The aggregate is the unweighted mean of the defined unit rates.

## SHACL

SHACL runs with `inference: none`, as in `docs/shacl_mapping.md`. One violation is one validation result.

A **generated individual** is a URI that is the subject of an `rdf:type` triple in the prefix-completed graph. This is the same set `n_individuals` counts.

`shacl_violations_per_individual` is the number of violations whose focus node is a generated individual, divided by `n_individuals`. It is null when the parse failed, when SHACL did not run, or when `n_individuals` is 0. The numerator is still stored when `n_individuals` is 0: it is 0, because no focus node can be a generated individual.

`shacl_other_focus_count` is the number of violations whose focus node is not a generated individual. Range shapes use the object as the focus node, so an object that was not declared with `rdf:type` is counted here and not in the per-individual rate. A generated individual that is also an object stays in the per-individual rate. The aggregate of the other-focus count is the unweighted mean of the counts, plus the sum. The count is null when SHACL did not run.

The repair loop feeds back every SHACL message, from both groups. The split above is only the metric.

## Inconsistent flag

`inconsistent` is true when HermiT finishes and reports the ontology plus the generated graph as inconsistent (`consistent` is false). It is false when HermiT finishes and reports consistent. It is null when the parse failed or when HermiT does not finish (`consistent` is null and `error` is set). A reasoner error is not coded as inconsistent.

The aggregate is the unweighted mean of the defined 0/1 values.

An inconsistent HermiT run raises `OwlReadyInconsistentOntologyError` and does not return a class list. The stored unsatisfiable-class tuple is empty in that case, and it is not read as "no unsatisfiable classes".

## New unsatisfiable classes

Before any individuals are added, HermiT is run on the ontology alone. That baseline list is the set of classes equivalent to `owl:Nothing`, excluding `owl:Nothing` itself.

`new_unsatisfiable_classes` is the sorted set difference: classes unsatisfiable with the generated graph, minus the baseline classes. Classes that were already unsatisfiable do not count. Classes that were unsatisfiable at baseline and are absent from a later consistent run are not a separate metric.

`new_unsatisfiable` is true when that difference is non-empty and false when it is empty. It is null when the individual run did not finish consistent, when the baseline run did not finish consistent, or when the parse failed. The null case includes an inconsistent individual run, because that run has no class list.

The aggregate is the unweighted mean of the defined 0/1 flags. The class IRIs are stored on the unit.

## Coverage

The relevant signature of a target class is:

- the target class itself
- every named class mentioned by `axioms_for_class` for that target
- every object or data property mentioned by those axioms

A relevant class is used when it is the object of an `rdf:type` triple in the generated graph. A relevant property is used when it is a predicate of the generated graph. A class used only as a non-type object is not counted as used.

`coverage_count` is the number of distinct relevant classes used plus the number of distinct relevant properties used. `n_relevant` is the size of the signature. `coverage_rate` is `coverage_count / n_relevant`. The signature always contains the target class, so `n_relevant` is at least 1 when the ontology loaded.

Both the count and the rate are null when the parse failed. The primary coverage metric is `coverage_count`, the number of distinct target-relevant names used. `coverage_rate` is reported beside it because signature sizes differ across classes. Both were defined before the run.

## Duplicate rate

Each generated individual has a description: the sorted list of its `(predicate, value)` pairs. A URI object that is itself a generated individual is replaced by that individual's description. A repeat encountered while following those links is the token `cycle`. A URI object that is not a generated individual stays as that IRI. A literal keeps its lexical form, datatype, and language.

`duplicate_rate` is `(n_individuals - n_distinct_descriptions) / n_individuals`. It is 0 when every description is unique. It is null when the parse failed or when `n_individuals` is 0.

The aggregate is the unweighted mean of the defined rates.

## Tokens

Token counts are the provider counts. For Ollama those are `prompt_eval_count` and `eval_count`. A missing count stays null and is not estimated.

On one completion the two counts are stored separately. On an A2 unit the condition's token total is the sum across rounds, including round 0. Round 0 is the cached A1 completion, so adding the A1 total to the A2 total counts those tokens twice. The experiment total used for cost therefore sums only completions with `cached` false.

The per-unit token metric for the bootstrap is the sum of uncached `prompt_eval_count` on that unit, and the same for `eval_count`. A unit whose uncached calls include a missing count is left out of that mean, and the exclusion is counted.

## Repair rounds

A0 and A1 do not repair. Their `n_repair_rounds` is 0.

A2 round 0 is the A1 request for the same unit. It must be a cache hit when A1 is run earlier in the same process. The config rejects an A2 condition that does not have A1 listed before it. Rounds 1, 2, and 3 are repair calls. `n_repair_rounds` is how many of those calls were made. The maximum is 3. The config cannot set a higher maximum.

The loop stops when the round is clean, or after round 3, whichever comes first. A round is clean when all of the following hold:

- `parse_ok` is true
- `undefined_terms` is empty
- HermiT finished with `consistent` true
- SHACL ran and returned zero violations

Clean does not look at the individual count, truncation, class-as-individual triples, coverage, or duplicates. Those are measured. They are not fed back, and they do not by themselves start another round.

The feedback after a round that is not clean, and only then, contains:

- the prefix-completed parse error, or `none`
- the OWL status: `none`, `inconsistent`, the reasoner error, or `not run` when the parse failed
- SHACL messages deduplicated by the exact message string, sorted, at most 10, plus a line with the number omitted when more than 10 distinct messages remain
- the undefined-term IRIs, or `none`
- the previous raw completion

The repair prompt is the A1 prompt, then that feedback, then the request for the full corrected Turtle. The wording is in `docs/prompt_versions.md`. Each repair call uses temperature 0, the fixed request seed, and the config token cap.

Every round stores its completion, its metric block, its validation report, whether it was clean, and the feedback text sent after it.

## Aggregates and comparisons

For a 0/1 metric the unit value is 1 or 0, or null when the definition above says the value is missing. The reported estimate is the unweighted mean of the non-null unit values.

The confidence interval is a percentile bootstrap of that mean over classes, from `ollab.core.stats.bootstrap_ci`: 9999 resamples, confidence 0.95, bootstrap RNG seed 0, linear quantile interpolation. The interval is stored with the estimate, the class count, the resample count, and the RNG seed.

Paired comparisons use the within-unit differences `A1 - A0`, `A2 - A0`, and `A2 - A1`. A unit enters a paired metric only when both conditions have a non-null value. The bootstrap is on the mean of those differences. The number of units dropped is stored. All three pairs are reported, including a negative mean.

Token sums and wall-clock totals are totals. They do not get a confidence interval. The per-unit means of uncached tokens, uncached LLM seconds, and validation seconds do.

## Wall-clock and the Phase 2b extrapolation

Wall-clock is `time.perf_counter` around the suite. It includes model calls, cache reads, HermiT, SHACL, and the process overhead of this runner.

These durations are stored separately:

- `llm_elapsed_s` around each provider call, including a cache read
- `validate_elapsed_s` around HermiT plus SHACL for each scored graph
- `baseline_elapsed_s` for the ontology-only HermiT run, once per ontology

Phase 2b is not given a class count or a sample count in this document. No USD price is attached to `qwen2.5:7b`. The extrapolation computed from a plumbing run is the set of measured rates below, and nothing else:

- uncached completions per unit, counting A0, A1, and the A2 repair calls, and not counting an A2 round-0 cache hit
- uncached prompt tokens per unit, and uncached completion tokens per unit
- uncached LLM seconds per unit
- validation seconds per unit, not including the ontology-only baseline
- baseline seconds for each ontology actually run

For a later grid of `G` units and `B` ontologies drawn from the same protocol, the same model, the same prompt version, and the same temperature, the estimate is `G` times the per-unit rates, plus the baseline seconds of those ontologies. The repair rate inside the token and time rates is the rate observed in the run that produced them. The class mix is the mix of that run. The estimate is not a price in dollars. A different model is not estimated by multiplying these rates.
