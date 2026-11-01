# OWL axioms to SHACL shapes

This mapping is deliberately strict. OWL domain and range axioms are semantic
inferences: if a property is used, a reasoner may infer the types of its
subject and object. SHACL treats the same triples as constraints that the data
must already satisfy. This project uses that strict reading. pyshacl is run
with `inference: none`, so an individual must carry an explicit `rdf:type`.
A missing type is a violation, not something the checker fills in.

Shape IRIs are `http://ollab.local/shapes/{axiom_id}` plus a suffix when one
axiom produces several shapes. Messages name the IRIs involved.

## Translated

| Asserted OWL | SHACL |
| --- | --- |
| `P rdfs:domain C`, `C` a named class | Node shape, `sh:targetSubjectsOf P`, `sh:class C` |
| `P rdfs:range C`, `C` a named class | Node shape, `sh:targetObjectsOf P`, `sh:class C` |
| `C rdfs:subClassOf (P min n)`, or `max`, or `exactly`, `C` named | Node shape, `sh:targetClass C`, property `sh:path P` with `sh:minCount` and/or `sh:maxCount` |
| The same with a named qualifier class | `sh:qualifiedMinCount` and/or `sh:qualifiedMaxCount`, `sh:qualifiedValueShape` / `sh:class` |
| `P` a `owl:FunctionalProperty` | Node shape, `sh:targetSubjectsOf P`, `sh:maxCount 1` |
| `P` a `owl:InverseFunctionalProperty` | Node shape, `sh:targetObjectsOf P`, `sh:path [ sh:inversePath P ]`, `sh:maxCount 1` |
| `A owl:disjointWith B`, both named | Two node shapes: `sh:targetClass A` / `sh:not [ sh:class B ]` and the converse |
| `owl:AllDisjointClasses` of named classes | The same pairwise `sh:not` shapes |
| `C owl:equivalentClass` a bare cardinality restriction, `C` named | The same count constraint as a subclass cardinality axiom. SHACL does not state the converse (every resource meeting the count is in `C`) |

`sh:minCount` and `sh:maxCount` for `exactly n` are both set to `n`.

## Not translated

These stay in `ShapeSet.untranslated` with a reason string:

- `rdfs:subClassOf` between named classes, or any subclass axiom whose superclass is not a cardinality restriction. Subclass inclusion is an inference, not a closed-world constraint, in this phase.
- `owl:someValuesFrom`, `owl:allValuesFrom`, and `owl:hasValue`.
- `owl:equivalentClass` other than a bare cardinality restriction. An equivalence is two directions; a SHACL shape only checks the individuals that already have the target type.
- Domain or range whose class expression is not a single named class.
- Qualified cardinality whose filler is not a named class.
- Disjointness that mentions a non-named class expression.
- Property characteristics other than functional and inverse-functional (transitive, symmetric, inverse, chains, keys).

Individuals are not validated against those axioms by the SHACL step. HermiT remains the consistency check.
