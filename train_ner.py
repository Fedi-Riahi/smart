import spacy
from spacy.training import Example
import random
from train_data import TRAIN_DATA

def train_ner_model(output_dir="ner_model", n_iter=20):
    # Load pre-trained model
    nlp = spacy.load("en_core_web_sm")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add SKILL label
    ner.add_label("SKILL")

    # Disable other pipes for efficiency
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")

    # Save model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_ner_model()
