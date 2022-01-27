# Import modules
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Select training data
train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey','rec.sport.baseball'], subset='train', shuffle=True, random_state=108)

# Explore dataset
#print(emails.target_names)

# Print emails at index 5
#print(emails.data[5])

# Print labels at index 5
#print(emails.target[5])

# Check label names
#print(emails.target_names)

# Select testing data
test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey','rec.sport.baseball'], subset='test', shuffle=True, random_state=108)

# Create a CountVectorizer object
counter = CountVectorizer()

# Train counter
counter.fit(test_emails.data + train_emails.data)

# Make a list of counts of words in the training data
train_counts = counter.transform(train_emails.data)

# Make a list of counts of words in the testing data
test_counts = counter.transform(test_emails.data)

# Create a MultinomialNB object
classifier = MultinomialNB()

# Train the model
classifier.fit(train_counts, train_emails.target)

# Print out the score on the test data
print(classifier.score(test_counts, test_emails.target))