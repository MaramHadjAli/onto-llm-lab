# Ontology licenses and provenance

The ontology files remain the property of their authors. This repository does not
claim a license for them and does not decide whether redistribution is allowed.
The notes below record what was found in the files and on the upstream pages.

The bytes were retrieved on 2026-11-01 and are otherwise unmodified.
SHA-256 values were computed with Python `hashlib.sha256` on those bytes.
`wine.rdf` and `food.rdf` are stored under `.owl` names; only the filenames differ.

## pizza.owl

- Upstream repository: <https://github.com/owlcs/pizza-ontology>
- Commit: `4922ecbdf5535a00a6515276324c3aa7f5e4407a`
- Commit page: <https://github.com/owlcs/pizza-ontology/commit/4922ecbdf5535a00a6515276324c3aa7f5e4407a>
- Commit date reported by the GitHub API: 2025-03-02T23:00:18Z
- Commit message: "Update pizza.owl to remove co-ode.owl"
- File URL at that commit: <https://raw.githubusercontent.com/owlcs/pizza-ontology/4922ecbdf5535a00a6515276324c3aa7f5e4407a/pizza.owl>
- Also published as: <https://raw.githubusercontent.com/owlcs/pizza-ontology/master/pizza.owl>
- Retrieved: 2026-11-01
- Bytes: 241414
- SHA-256: `0de2cd4df0bc09dfd9403df46ae7aa9d26ec31c32cc9a59c514f459c312a751c`
- The SHA-256 of the file at the commit URL above was computed again on 2026-11-01 and matches the stored file.
- `rdfs:comment` in the file: "An example ontology that contains all constructs required for the various versions of the Pizza Tutorial run by Manchester University"
- `owl:versionInfo` values in the file:
  - "v.1.4. Added Food class (used in domain/range of hasIngredient), Added several hasCountryOfOrigin restrictions on pizzas, Made hasTopping invers functional"
  - "version 1.5"
  - "v.1.5. Removed protege.owl import and references. Made ontology URI date-independent"
- License: none found. The GitHub API reported `"license": null` for `owlcs/pizza-ontology`. `https://raw.githubusercontent.com/owlcs/pizza-ontology/master/LICENSE` returned HTTP 404. The OWL file itself contains no copyright or license statement. This repository does not add one.

## wine.owl

- Upstream document: OWL Web Ontology Language Guide, W3C Recommendation 10 February 2004, <https://www.w3.org/TR/2004/REC-owl-guide-20040210/>
- File URL: <https://www.w3.org/TR/2004/REC-owl-guide-20040210/wine.rdf>
- Stored as: `wine.owl`
- Retrieved: 2026-11-01
- Bytes: 78225
- SHA-256: `bb1666593f9d7ba2ec9cb06dd4fcc17751ac8d523dc33ce22bcee6beeda778eb`
- Copyright notice displayed on that Recommendation: "Copyright © 2004 W3C® (MIT, ERCIM, Keio), All Rights Reserved. W3C liability, trademark, document use and software licensing rules apply."
- Links printed with that notice include <https://www.w3.org/Consortium/Legal/ipr-notice#Copyright> and <https://www.w3.org/Consortium/Legal/copyright-documents> (document use). Fetched on 2026-11-01, the document-use URL redirected to <https://www.w3.org/copyright/document-license-2023/> ("Document license - 2023 version"). This file records the redirect; it does not claim the 2004 text and the 2023 text are the same.
- The file's `rdfs:comment` says it was derived from the DAML Wine ontology at <http://ontolingua.stanford.edu/doc/chimaera/ontologies/wines.daml>.
- The file declares `owl:imports` of `http://www.w3.org/TR/2003/PR-owl-guide-20031209/food`.
- A search of the file for "copyright" and "license" found no matches. The file itself has no license statement.

## food.owl

`wine.owl` imports the food ontology, and `food.owl` imports wine. The OWL Guide examples are a circular import. `food.owl` is vendored so a later offline reasoner does not have to fetch it.

- File URL: <https://www.w3.org/TR/2004/REC-owl-guide-20040210/food.rdf>
- Stored as: `food.owl`
- Retrieved: 2026-11-01
- Bytes: 45326
- SHA-256: `64af31d99ee62f58b7daa18977bbb27973c67a717d5d6317f42400bd53504184`
- Same W3C Recommendation notice as `wine.owl`.
- The file's `rdfs:comment` says it was derived from the same DAML Wine ontology, substantially modified.
- The file declares `owl:imports` of `http://www.w3.org/TR/2003/PR-owl-guide-20031209/wine`.
- A search of the file for "copyright" and "license" found no matches. The file itself has no license statement. The W3C notice quoted under `wine.owl` is the notice on the Recommendation page, not text inside `food.owl`.
