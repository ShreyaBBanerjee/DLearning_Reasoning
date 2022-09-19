# DLearning_Reasoning

This project investigates how well a AI system can learn to perform reasoning tasks, from scratch. A variant of Convolutional Neural Network (CNN) model is designed and implemented on datasets of propositional logic problems. 

Assume that we have a formal logic F ≡ ⟨L, I , ⊥⟩, where L is the language of the formal logic, I is the inference system and ⊥ ∈ L. Any reasoning problem Γ ⊢ φ can be posed as a consistency problem if certain conditions aare satisfied by I. We reformulate the reasoning problem to classification task of consistency checking.

The original dataset consists of sets of formulae in Conjunctive Normal Form (CNF). Each formula φ in a set is in the form of a disjunction. Each literal l of the disjunction is an atom P or its negation. We generate a radom set of formulae Γ by randomly generating u clauses, where each clause has v random literals drawn from a set of w atomic propositions. For each such random set of sentences, we run a theorem prover to check whether the sentence is consistent or not. Here we test on two distinct datasets for propositional logic with u,v,w = 4 and u,v,w = 5.

The 4X4 and 5X5 datasets consists of 45000 samples each. We do 5-fold crosss validation, with 
