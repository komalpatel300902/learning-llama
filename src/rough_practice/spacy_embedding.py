import spacy
import numpy as np

# Load spaCy's medium English model (includes word vectors)
nlp = spacy.load("en_core_web_md")  # Run `python -m spacy download en_core_web_md` if not installed

# Sample sentence
sentence = "I love AI"

# Process sentence

doc = nlp(sentence)
print(doc)

# Extract token vectors
input_embeddings = np.array([token.vector for token in doc])

print("Tokens:", [token.text for token in doc])
print("Shape:", input_embeddings.shape)  # e.g., (3, 300)
print("Embeddings:\n", input_embeddings)
