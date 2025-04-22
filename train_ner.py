import spacy
from spacy.training import Example
import random
from train_data import TRAIN_DATA

def train_ner_model(output_dir="ner_model", n_iter=30):
    # Option 1: Start with blank model (recommended for custom entity only)
    nlp = spacy.blank("en")
    
    # Option 2: Or if you want to use pre-trained model with disabled entities:
    # nlp = spacy.load("en_core_web_sm")
    # for entity_type in ["DATE", "GPE", "CARDINAL", "ORG", "PERSON", "NORP", "TIME"]:
    #     nlp.entity.remove_label(entity_type)
    
    # Add NER pipeline
    ner = nlp.add_pipe("ner")
    
    # Add only your SKILL label
    ner.add_label("SKILL")
    
    # Disable other pipes if using pre-trained model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                # Create example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update model
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")
    
    # Save model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_ner_model()