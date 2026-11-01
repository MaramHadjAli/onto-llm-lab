"""Exact template sentences for the mini ontology."""

from ollab.core.ontology import load_ontology
from ollab.core.verbalize import local_name, verbalize
from tests.support import MINI, ONTOLOGIES

EXPECTED = {
    "Every Named pizza is a Pizza.",
    "Every Cheese topping is a Pizza topping.",
    "Every Meat topping is a Pizza topping.",
    "Every Cheesy pizza relates via has topping to some Cheese topping.",
    "Every Cheese only pizza relates via has topping to only Cheese topping.",
    "Every Interesting pizza relates via has topping to at least 3 things.",
    "Every Closed pizza relates via has topping to at most 2 Meat topping.",
    "Every Exact pizza relates via has topping to exactly 1 Cheese topping.",
    "Every American pizza relates via has country to the value America.",
    "Every Flexible pizza is (has topping some (Cheese topping or Meat topping)).",
    "Every Odd is (unsupported-expression).",
    "Mozzarella topping is equivalent to Cheese topping.",
    "Vegetarian pizza is equivalent to (Pizza and has topping only Cheese topping).",
    "Three topping pizza is equivalent to has topping exactly 3.",
    "White pizza is equivalent to (Pizza and has topping value Mozzarella topping).",
    "Pizza is disjoint with Pizza topping.",
    "Cheese topping is disjoint with Meat topping.",
    "Pizza, Pizza topping and Pizza base are pairwise disjoint.",
    "has topping has domain Pizza.",
    "has base has domain Pizza.",
    "has calories has domain Pizza.",
    "has topping has range Pizza topping.",
    "has base has range Pizza base.",
    "has calories has range integer.",
    "has base is functional.",
    "has calories is functional.",
    "is base of is inverse-functional.",
}


def test_mini_sentences_match_the_templates() -> None:
    ontology = load_ontology(MINI)
    labels = ontology.labels()
    sentences = {verbalize(axiom, labels) for axiom in ontology.axioms()}
    assert sentences == EXPECTED


def test_public_verbalisations_are_nonempty_sentences() -> None:
    for name in ("pizza.owl", "wine.owl", "food.owl"):
        ontology = load_ontology(ONTOLOGIES / name)
        labels = ontology.labels()
        for axiom in ontology.axioms():
            text = verbalize(axiom, labels)
            assert text.endswith(".")
            assert text.strip() == text
            assert "  " not in text


def test_pizza_subclass_uses_label_and_local_name() -> None:
    ontology = load_ontology(ONTOLOGIES / "pizza.owl")
    labels = ontology.labels()
    base = "https://raw.githubusercontent.com/owlcs/pizza-ontology/refs/heads/master/pizza.owl#"
    found = [
        verbalize(axiom, labels)
        for axiom in ontology.axioms()
        if axiom.left is not None
        and axiom.left.iri == base + "Pizza"
        and axiom.right is not None
        and axiom.right.iri == base + "Food"
    ]
    assert found == ["Every Pizza is a Food."]


def test_local_name_splits_on_hash_or_slash() -> None:
    assert local_name("http://example.org/mini#Pizza") == "Pizza"
    assert local_name("http://example.org/ontologies/wine") == "wine"
