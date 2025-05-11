import spacy
nlp = spacy.load("en_core_web_md")

word1 = nlp.vocab["king"].vector
word2 = nlp.vocab["queen"].vector
word3 = nlp.vocab["apple"].vector
word4 = nlp.vocab["good"].vector
word5 = nlp.vocab["bad"].vector


print("Similarity king-queen:", nlp.vocab["king"].similarity(nlp.vocab["queen"]))
print("Similarity king-apple:", nlp.vocab["king"].similarity(nlp.vocab["apple"]))
print("Similarity : ", nlp.vocab["goob"].similarity(nlp.vocab["bad"]))