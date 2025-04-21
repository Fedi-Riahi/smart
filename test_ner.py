import spacy

def test_ner_model(model_dir="ner_model"):
    nlp = spacy.load(model_dir)
    test_texts = [
        "Proficient in Python, JavaScript, and TypeScript programming.",
        "Experienced with Ruby, Django, and MySQL for web development.",
        "Built ML models using TensorFlow, PyTorch, and machine learning.",
        "Deployed apps on AWS with Docker and Kubernetes.",
        "Followed Agile and Scrum methodologies with Git and Kanban.",
        "Expert in Node.js, RESTful APIs, and GraphQL services.",
        "Developed using .NET Core and Entity Framework.",
        "Skilled in machine learning and deep learning techniques."
    ]
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        print(f"Skills: {skills}")

if __name__ == "__main__":
    test_ner_model()
