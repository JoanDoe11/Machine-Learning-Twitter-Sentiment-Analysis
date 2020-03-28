# Machine-Learning-Twitter-Sentiment-Analysis
Machine Learning: implementation of the Naive Bayes algorithm for creating a mathematical model for evaluating tweets as positive or negative.

# Project

Create a model that can distinguish a tweet as positive (class: 1) or negative (class: 0), using the Naive Bayes Classification algorithm. Data is unstructured - before the classification algorithm, it needs to be processed.

Data: data was given by Racunarski fakultet, for the purpose of this project. Therefore, I cannot make it public.

# Machine learning

Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions. Machine learning algorithms build a mathematical model based on sample data (training data) in order to make predictions or decisions without being explicitly programmed to perform the task.

# Models for classification

Classifiers are models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set.

Probability model is a model that assigns probabilities of classes to problem instances.

input: `x = (x1,...,xn)` - problem instance that consists of n features

output: `p(Ck|x1,...,xn)` - probability that this instance is a member of Ck class

# Naive Bayes

Naive Bayes classifiers are a family of simple probabilistic classifiers baseed on applying Bayes theorem with strong independence assumptions between the features. It is a popular method for text categorization, the problem of judging documents as belonging to one category or the other with word frequencies as the features.
All naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.

Bayes' theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. It gives us the probability that an event A will happen, if we know that event B has already happened.

`P(A|B) = [ P(B|A)*P(A)]/P(B)` , where A and B are events, and P(B) != 0.

Applied to the conditional probability model, Bayes looks like this:

`p(Ck|x) = [p(Ck)*p(x|Ck)]/p(x)`

In practice, the denominator is constant, so the value for the probability depends on the numerator, which can be written as:

`p(Ck,x1,...,xn) = p(x1,...,xn,Ck) = p(x1|x2,...,Ck)*p(x2,...,xn,Ck) = ... = p(x1|x2,..,Ck)*p(x2|x3,...,Ck)*...*p(xn|Ck)*p(Ck)`

Naive part: we assume that all features in x are mutually independent, conditional on the category Ck. Therefore:

`p(xi|x1,...,xn, Ck) = p(xi|Ck)`

`p(Ck|x1,...,xn)` is proportional to `p(Ck)*p(x1|Ck)*..*p(xn|Ck)` which we can scale with:

`evidence: Z = p(x) = p(C1)*p(x|C1) + p(C2)*p(x|C2) * ...`

To conclude, the Naive Bayes probability model is denoted as follows:

`p(Ck|x1,..,xn) = 1/Z * p(Ck) * p(x1|Ck) * ... * p(xn|Ck)`

The Naive Bayes Classifier combines this model with a decision rule, which is often to pick the hypothesis that is most probable (maximum a posteriori decision rule) - pick the Ck for which the probability is the highest.

# initialization

We can calculate the initial values for p(Ck) in two ways:
* assuming equiprobable classes - p(Ck) = 1 / #classes
* calculating an estimate for the class probability from the training set - p(Ck) = |Ck| / #samples

# NLP

Natural Language Processing is a field in machine learning with the ability of a computer to understand, analyze, manipulate and potentually generate human language.

While reading data, we get data in the structured or unstructured format. A structured format has a wel defined pattern whereas unstructured data has no proper structure.

Cleaning up the text data is necessary to highlight attributes that we're going to want our machine learning system to pick up on. It usually consists of removing punctuation, tokenization (separating texts into untis - sentences/words), and removing stopwords (the, is, in, at, for...).

Stemming helps reduce a word to its stem form because it often makes sense to treat related words in the same way. It removes suffices (-ing, -ly..). Lemmatizing derives the root form of a word. It's more axxurate than stemming as it uses a dictionary-based approach - the morphological analysis to the root word, but stemming is faster than lemmatizing because it simply chops off the end of the word.

Vectorizing data is the process of encoding text to a numeric form in order to create feature vectors, so that machine learning algorithms can understand the data.
