import spacy
from pathlib import Path
import logging
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def estimate_eta(step: str, data_size: int = 0) -> float:
    """Estimate ETA in seconds for each step based on step type and data size."""
    if step == "test":
        return 3  # Few test texts, very quick
    return 0

def test_model(nlp: spacy.language.Language, test_texts: List[str]) -> None:
    """Test the model on new texts and print detected entities."""
    for text in test_texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "SKILL"]
        logger.info(f"Text: {text}")
        logger.info(f"Entities: {entities}")

def main():
    # Load the trained model
    model_path = Path("cvApp/ner_model")
    if not model_path.exists():
        logger.error(f"Model directory {model_path} does not exist. Please train the model first.")
        return

    try:
        nlp = spacy.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return

    # Step 6: Test the model
    logger.info(f"Step 6: Testing model. ETA: {estimate_eta('test'):.2f} seconds")
    start_time = time.time()
    test_texts = [
        "Implemented CI/CD pipelines using Jenkins and GitLab CI.",
        "Managed infrastructure with Terraform and Ansible.",
        "Containerized applications using Docker and Kubernetes.",
        "Monitored systems with Prometheus and Grafana.",
        "Automated deployments using Bash and Python scripts."
    ]
    test_model(nlp, test_texts)
    duration = time.time() - start_time
    logger.info(f"Step 6 completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()
