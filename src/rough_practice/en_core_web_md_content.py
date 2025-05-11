import spacy

nlp = spacy.load("en_core_web_md")
vector = nlp.vocab["apple"].vector
print("Vector shape:", vector.shape)
print("First 5 dims:", vector[:5])
