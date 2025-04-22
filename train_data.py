TRAIN_DATA = [
    # Programming Languages
    ("Skilled in Python, Java, and C++.", {
        "entities": [(11, 17, "SKILL"), (19, 23, "SKILL"), (29, 32, "SKILL")]  # Python, Java, C++
    }),
    ("Proficient in JavaScript, TypeScript, and Go.", {
        "entities": [(14, 24, "SKILL"), (26, 36, "SKILL"), (42, 44, "SKILL")]  # JavaScript, TypeScript, Go
    }),
    ("Experience with Ruby, PHP, and Swift.", {
        "entities": [(16, 20, "SKILL"), (22, 25, "SKILL"), (31, 36, "SKILL")]  # Ruby, PHP, Swift
    }),
    ("Worked with R, MATLAB, and Scala.", {
        "entities": [(12, 13, "SKILL"), (15, 21, "SKILL"), (27, 32, "SKILL")]  # R, MATLAB, Scala
    }),

    # Frameworks & Libraries
    ("Developed apps using React, Angular, and Vue.js.", {
        "entities": [(21, 26, "SKILL"), (28, 35, "SKILL"), (41, 47, "SKILL")]  # React, Angular, Vue.js
    }),
    ("Built APIs with Django, Flask, and Spring Boot.", {
        "entities": [(16, 22, "SKILL"), (24, 29, "SKILL"), (35, 46, "SKILL")]  # Django, Flask, Spring Boot
    }),
    ("Used TensorFlow, PyTorch, and Keras for ML projects.", {
        "entities": [(5, 15, "SKILL"), (17, 24, "SKILL"), (30, 35, "SKILL")]  # TensorFlow, PyTorch, Keras
    }),

    # Databases & DevOps
    ("Managed databases like MySQL, PostgreSQL, and MongoDB.", {
        "entities": [(23, 28, "SKILL"), (30, 40, "SKILL"), (46, 53, "SKILL")]  # MySQL, PostgreSQL, MongoDB
    }),
    ("Deployed apps on AWS, Azure, and Google Cloud.", {
        "entities": [(17, 20, "SKILL"), (22, 27, "SKILL"), (33, 45, "SKILL")]  # AWS, Azure, Google Cloud
    }),
    ("Implemented CI/CD using Jenkins, Docker, and Kubernetes.", {
        "entities": [(24, 31, "SKILL"), (33, 39, "SKILL"), (45, 55, "SKILL")]  # Jenkins, Docker, Kubernetes
    }),

    # Methodologies & Tools
    ("Followed Agile, Scrum, and Kanban methodologies.", {
        "entities": [(9, 14, "SKILL"), (16, 21, "SKILL"), (27, 33, "SKILL")]  # Agile, Scrum, Kanban
    }),
    ("Used Git, Jira, and VS Code for development.", {
        "entities": [(5, 8, "SKILL"), (10, 14, "SKILL"), (20, 27, "SKILL")]  # Git, Jira, VS Code
    }),

    # Edge Cases & Variations
    ("Expert in Node.js and Express framework.", {
        "entities": [(10, 17, "SKILL"), (22, 29, "SKILL")]  # Node.js, Express
    }),
    ("Familiar with .NET Core and Entity Framework.", {
        "entities": [(14, 23, "SKILL"), (28, 44, "SKILL")]  # .NET Core, Entity Framework
    }),
    ("Knowledge of machine learning and deep learning.", {
        "entities": [(13, 29, "SKILL"), (34, 47, "SKILL")]  # machine learning, deep learning
    }),
    ("Worked on RESTful APIs and GraphQL.", {
        "entities": [(10, 22, "SKILL"), (27, 34, "SKILL")]  # RESTful APIs, GraphQL
    }),

    # Personal Information
    ("Contact me at john.doe@email.com or (123) 456-7890", {
        "entities": []
    }),
    ("Located in New York, NY from January 2020 to present", {
        "entities": []
    }),

    # Numbers and Quantities
    ("Managed budget of $500,000 annually", {
        "entities": []
    }),
    ("Led team of 15 developers across 3 locations", {
        "entities": []
    }),

    # Job Titles and General Terms
    ("Senior Software Engineer at Tech Company Inc.", {
        "entities": []
    }),
    ("Responsible for product development lifecycle", {
        "entities": []
    }),

    # Dates and Time Periods
    ("Worked from Q1 2019 to Q3 2022", {
        "entities": []
    }),
    ("Project completed in 6 months", {
        "entities": []
    }),
]

def validate_train_data(data):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    for text, annot in data:
        doc = nlp.make_doc(text)
        # Print tokens for debugging
        print(f"\nText: '{text}'")
        print("Token | Index | Span")
        print("-" * 30)
        for token in doc:
            print(f"{token.text:<12} | {token.idx:<5} | ({token.idx}, {token.idx + len(token.text)})")
        # Validate entities
        for start, end, label in annot["entities"]:
            assert 0 <= start < end <= len(text), f"Invalid entity in '{text}': {start, end, label}"
            assert text[start:end].strip(), f"Empty entity in '{text}': {start, end, label}"
            # Check if span starts and ends at token boundaries
            start_found = False
            end_found = False
            for token in doc:
                if token.idx == start:
                    start_found = True
                if token.idx + len(token.text) == end:
                    end_found = True
            if not start_found:
                print(f"Warning: Entity ({start}, {end}, '{label}') in '{text}' does not start at a token boundary")
            if not end_found:
                print(f"Warning: Entity ({start}, {end}, '{label}') in '{text}' does not end at a token boundary")
        from spacy.training import offsets_to_biluo_tags
        tags = offsets_to_biluo_tags(doc, annot["entities"])
        if '-' in tags:
            print(f"Misaligned entities in '{text}': {tags} | Entities: {annot['entities']}")
            for i, (token, tag) in enumerate(zip(doc, tags)):
                if tag == '-':
                    print(f"  Token '{token.text}' at index {token.idx} is misaligned")
            raise AssertionError(f"Misaligned entities detected in '{text}'")
        print(f"Validated text: '{text}'")
    print("All data validated successfully.")

def generate_entity_spans(text, skills, label="SKILL"):
    """Generate entity spans for given skills in text, ensuring token alignment."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp.make_doc(text)
    entities = []
    for skill in skills:
        # Find skill in text
        start_idx = text.find(skill)
        if start_idx == -1:
            print(f"Skill '{skill}' not found in text: '{text}'")
            continue
        end_idx = start_idx + len(skill)
        # Find tokens that cover the skill
        skill_tokens = [token for token in doc if token.idx >= start_idx and token.idx < end_idx]
        if not skill_tokens:
            print(f"Skill '{skill}' at {start_idx}-{end_idx} does not align with tokens in '{text}'")
            continue
        # Check if skill spans whole tokens
        if skill_tokens[0].idx == start_idx and skill_tokens[-1].idx + len(skill_tokens[-1].text) == end_idx:
            entities.append((start_idx, end_idx, label))
        else:
            print(f"Skill '{skill}' at {start_idx}-{end_idx} is misaligned with tokens: {[t.text for t in skill_tokens]}")
    return {"entities": sorted(entities)}  # Sort for consistency

if __name__ == "__main__":
    validate_train_data(TRAIN_DATA)
    # Example usage of generate_entity_spans
    # text = "Knowledge of machine learning and deep learning."
    # skills = ["machine learning", "deep learning"]
    # print(generate_entity_spans(text, skills))
