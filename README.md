# DLearning_Reasoning

This project investigates how well a AI system can learn to perform reasoning tasks, from scratch. A variant of Convolutional Neural Network, named Logical CNN (LCNN) model is designed and implemented on datasets of propositional logic problems. 

Assume that we have a formal logic F ≡ ⟨L, I , ⊥⟩, where L is the language of the formal logic, I is the inference system and ⊥ ∈ L. Any reasoning problem Γ ⊢ φ can be posed as a consistency problem if certain conditions aare satisfied by I. We reformulate the reasoning problem to classification task of consistency checking.

The original dataset consists of sets of formulae in Conjunctive Normal Form (CNF). Each formula φ in a set is in the form of a disjunction. Each literal l of the disjunction is an atom P or its negation. We generate a radom set of formulae Γ by randomly generating u clauses, where each clause has v random literals drawn from a set of w atomic propositions. For each such random set of sentences, we run a theorem prover to check whether the sentence is consistent or not. Here we test on two distinct datasets for propositional logic with u,v,w = 4 and u,v,w = 5.

The 4X8 and 5X10 datasets consists of 10K samples each. We do a split with 75000 training data and 25000 test data. To implement a CNN model on this data, the samples are preprocessed into matrix form. LCNN is designed to have 1 convolutional layer, ReLu activation function, 1 pooling layer followed by a fully connected layer. The output is fed into a standard 2-way softmax in order to perform the classification task. We use Adam optimizer.

The model is able to predict with ~93% accuracy for 4x8 and ~83% on 5x10 datasets, approximately. With lesser data the model overfits badly. The best results are obtained by running 7500 epochs of training with a batch size 10 and learning rate of 0.0005. As we note in the given figure, the loss converges steadily with each epoch while accuracy peaks to ~93% on 4x8 dataset.

The LCNN model is a novel DL architechture that learns reasoning task but only to some extent. It is biased towards syntactic structure and doesn't take full account of implicit semantic information. A hybrid architecture to learn the semantic information is suggested.
