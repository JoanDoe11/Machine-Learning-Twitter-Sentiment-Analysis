# Machine-Learning-Twitter-Sentiment-Analysis
Machine Learning: implementation of the Naive Bayes algorithm for creating a mathematical model for evaluating tweets as positive or negative.

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
