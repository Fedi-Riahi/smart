import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from spacy.scorer import Scorer
import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Expanded list of predefined skills (adjusted to reduce substring overlaps)
SKILLS = [
    # Programming Languages
    "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Ruby", "PHP", "Swift", "R",
    "MATLAB", "Scala", "Kotlin", "Rust", "Perl", "C#", "Objective-C", "Shell", "PowerShell",
    "SQL", "HTML", "CSS", "Dart", "Lua", "VBA",

    # Frameworks and Libraries
    "React", "Angular", "Vue.js", "Django", "Flask", "Spring Boot", "Node.js", "Express",
    ".NET Core", "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy", "Scikit-learn",
    "FastAPI", "Ruby on Rails", "Laravel", "Symfony", "Bootstrap", "jQuery", "D3.js",
    "OpenCV", "Hadoop", "Spark",

    # Databases and Cloud Platforms
    "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "SQL Server", "Redis", "Cassandra",
    "Elasticsearch", "DynamoDB", "Snowflake", "BigQuery", "AWS", "Azure", "Google Cloud",
    "Firebase", "Heroku", "Salesforce", "SAP",

    # DevOps and Tools
    "Jenkins", "Docker", "Kubernetes", "Terraform", "Ansible", "Git", "GitHub", "GitLab",
    "Bitbucket", "Jira", "Trello", "Confluence", "Prometheus", "Grafana", "Nagios",
    "CircleCI", "Travis CI", "Bamboo", "VS Code", "IntelliJ IDEA", "Eclipse", "PyCharm",

    # Methodologies and Processes
    "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "TDD", "BDD", "Lean", "Six Sigma",
    "project management", "risk management", "change management",

    # Hospitality Systems
    "Micros", "Hilton OnQ", "Opera PMS", "Fidelio", "Holidex", "Sabre", "Amadeus",

    # Data Science and Analytics
    "machine learning", "deep learning", "data analysis", "data visualization", "statistics",
    "Power BI", "Tableau", "Excel", "R Studio", "Jupyter", "Matplotlib", "Seaborn",

    # Business and HR Skills
    "Customer Service", "medical billing", "advertising", "marketing", "payroll",
    "employee relations", "performance reviews", "recruitment", "training",
    "compensation management", "benefits administration", "labor relations",
    "public relations", "financial management", "budgeting", "accounting",

    # Web and Digital Marketing
    "SEO", "SEM", "Google Analytics", "Facebook Ads", "WordPress", "Shopify",
    "content management", "digital marketing", "social media marketing",

    # Other Technical Skills
    "SharePoint", "RESTful APIs", "GraphQL", "SOAP", "Linux", "Unix", "Windows Server",
    "network administration", "cybersecurity", "penetration testing", "VMware", "Hyper-V"
]

def estimate_eta(step: str, data_size: int = 0) -> float:
    """Estimate ETA in seconds for each step based on step type and data size."""
    if step == "extract_csv":
        return 0.5  # Fast CSV reading
    elif step == "create_train_data":
        return max(5, data_size * 0.2)  # ~0.2s per resume
    elif step == "validate_data":
        return max(2, data_size * 0.1)  # ~0.1s per resume
    elif step == "train_model":
        return max(300, data_size * 10)  # ~10s per resume, min 5min
    elif step == "evaluate_model":
        return 5  # Small subset, quick
    elif step == "test_model":
        return 3  # Few test texts, very quick
    return 0

def extract_resume_texts_from_csv(csv_path: str = "resumes/Resume.csv") -> List[str]:
    """Extract resume texts from the 'Resume_str' column of a CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file {csv_path} does not exist.")
        return []

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'Resume_str' not in df.columns:
            logger.error("CSV file must contain a 'Resume_str' column.")
            return []

        # Filter out empty or invalid texts
        resume_texts = [str(text) for text in df['Resume_str'] if pd.notna(text) and str(text).strip()]
        if not resume_texts:
            logger.warning("No valid resume texts found in CSV.")
        else:
            logger.info(f"Extracted {len(resume_texts)} resume texts from {csv_path}")
        return resume_texts

    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return []

def generate_entity_spans(text: str, skills: List[str] = SKILLS, label: str = "SKILL") -> Dict[str, Any]:
    """Generate non-overlapping entity spans for given skills in text, ensuring token alignment."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp.make_doc(text)
    entities = []

    # Sort skills by length (descending) to prioritize longer matches
    sorted_skills = sorted(skills, key=len, reverse=True)

    for skill in sorted_skills:
        start_idx = 0
        while start_idx != -1:
            start_idx = text.lower().find(skill.lower(), start_idx)
            if start_idx == -1:
                break
            end_idx = start_idx + len(skill)
            # Verify token alignment
            skill_tokens = [t for t in doc if t.idx >= start_idx and t.idx < end_idx]
            if skill_tokens and skill_tokens[0].idx == start_idx and skill_tokens[-1].idx + len(skill_tokens[-1].text) == end_idx:
                # Check for overlap with existing entities
                overlaps = any(
                    (e_start <= start_idx < e_end) or (start_idx <= e_start < end_idx)
                    for e_start, e_end, _ in entities
                )
                if not overlaps:
                    entities.append((start_idx, end_idx, label))
                else:
                    logger.debug(f"Skipped overlapping skill '{skill}' at {start_idx}-{end_idx} in text: '{text[start_idx:end_idx]}'")
            else:
                logger.debug(f"Skill '{skill}' at {start_idx}-{end_idx} misaligned in text: '{text[:50]}...'")
            start_idx += 1

    return {"entities": sorted(entities, key=lambda x: x[0])}

def validate_train_data(data: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Validate training data to ensure entity spans are correctly aligned and non-overlapping."""
    nlp = spacy.load("en_core_web_sm")
    for text, annot in data:
        doc = nlp.make_doc(text)
        for start, end, label in annot["entities"]:
            if not (0 <= start < end <= len(text)):
                raise ValueError(f"Invalid entity in '{text[:50]}...': {start, end, label}")
            if not text[start:end].strip():
                raise ValueError(f"Empty entity in '{text[:50]}...': {start, end, label}")
            # Check token boundaries
            start_found = any(token.idx == start for token in doc)
            end_found = any(token.idx + len(token.text) == end for token in doc)
            if not start_found or not end_found:
                logger.warning(f"Entity ({start}, {end}, '{label}') in '{text[:50]}...' may not align with token boundaries")

        from spacy.training import offsets_to_biluo_tags
        tags = offsets_to_biluo_tags(doc, annot["entities"])
        if '-' in tags:
            raise ValueError(f"Misaligned entities in '{text[:50]}...': {tags} | Entities: {annot['entities']}")

    logger.info("All training data validated successfully")

def create_train_data(resume_texts: List[str], skills: List[str] = SKILLS) -> List[Tuple[str, Dict[str, Any]]]:
    """Create training data by generating entity annotations for resume texts."""
    train_data = []
    for text in resume_texts:
        text = " ".join(text.split())  # Normalize whitespace
        if not text:
            continue
        annotations = generate_entity_spans(text, skills)
        if annotations["entities"]:
            train_data.append((text, annotations))

    logger.info(f"Created {len(train_data)} training examples")
    return train_data

def train_ner_model(
    train_data: List[Tuple[str, Dict[str, Any]]],
    output_dir: str = "cvApp/ner_model",
    epochs: int = 50,
    batch_size: int = 8
) -> spacy.language.Language:
    """Train a NER model to recognize skills."""
    nlp = spacy.load("en_core_web_md")

    # Add or get NER pipe
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    # Add SKILL label
    for _, annotations in train_data:
        for ent in annotations['entities']:
            if ent[2] not in ner.labels:
                ner.add_label(ent[2])

    # Disable other pipes
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()
        for epoch in range(epochs):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=batch_size)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
            logger.info(f"Epoch {epoch + 1}, Losses: {losses}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    logger.info(f"Model saved to {output_dir}")
    return spacy.load(output_dir)

def evaluate_model(
    nlp: spacy.language.Language,
    test_data: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Evaluate the trained NER model."""
    scorer = Scorer()
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        scorer.update([example])
    logger.info(f"Evaluation scores: {scorer.scores}")
    return scorer.scores

def test_model(nlp: spacy.language.Language, test_texts: List[str]) -> None:
    """Test the model on new texts and print detected entities."""
    for text in test_texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "SKILL"]
        logger.info(f"Text: {text}")
        logger.info(f"Entities: {entities}")

def main():
    # Step 1: Extract resume texts from CSV
    logger.info(f"Step 1: Extracting resume texts from CSV. ETA: {estimate_eta('extract_csv'):.2f} seconds")
    start_time = time.time()
    resume_texts = extract_resume_texts_from_csv(csv_path="resumes/Resume.csv")
    duration = time.time() - start_time
    logger.info(f"Step 1 completed in {duration:.2f} seconds")

    if not resume_texts:
        logger.error("No resume texts extracted. Exiting.")
        return

    # Step 2: Create training data
    logger.info(f"Step 2: Creating training data. ETA: {estimate_eta('create_train_data', len(resume_texts)):.2f} seconds")
    start_time = time.time()
    train_data = create_train_data(resume_texts)
    duration = time.time() - start_time
    logger.info(f"Step 2 completed in {duration:.2f} seconds")

    if not train_data:
        logger.error("No training data generated. Exiting.")
        return

    # Step 3: Validate training data
    logger.info(f"Step 3: Validating training data. ETA: {estimate_eta('validate_data', len(train_data)):.2f} seconds")
    start_time = time.time()
    try:
        validate_train_data(train_data)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        return
    duration = time.time() - start_time
    logger.info(f"Step 3 completed in {duration:.2f} seconds")

    # Step 4: Train the model
    logger.info(f"Step 4: Training NER model. ETA: {estimate_eta('train_model', len(train_data)):.2f} seconds")
    start_time = time.time()
    trained_nlp = train_ner_model(train_data, output_dir="cvApp/ner_model")
    duration = time.time() - start_time
    logger.info(f"Step 4 completed in {duration:.2f} seconds")

    # Step 5: Evaluate the model
    logger.info(f"Step 5: Evaluating model. ETA: {estimate_eta('evaluate_model'):.2f} seconds")
    start_time = time.time()
    evaluate_model(trained_nlp, train_data[:min(5, len(train_data))])
    duration = time.time() - start_time
    logger.info(f"Step 5 completed in {duration:.2f} seconds")

    # Step 6: Test the model
    logger.info(f"Step 6: Testing model. ETA: {estimate_eta('test_model'):.2f} seconds")
    start_time = time.time()
    test_texts = [
        "Implemented CI/CD pipelines using Jenkins and GitLab CI.",
        "Managed infrastructure with Terraform and Ansible.",
        "Containerized applications using Docker and Kubernetes.",
        "Monitored systems with Prometheus and Grafana.",
        "Automated deployments using Bash and Python scripts."
    ]
    test_model(trained_nlp, test_texts)
    duration = time.time() - start_time
    logger.info(f"Step 6 completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()
