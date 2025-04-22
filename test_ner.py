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
    "Skilled in machine learning and deep learning techniques.",
    "Strong expertise in Java, Kotlin, and Android app development.",
    "Developed high-performance applications in C++ and Rust.",
    "Worked with Go (Golang) for scalable backend microservices.",
    "Experience in functional programming with Haskell and Scala.",
    "Built automation scripts using Bash and PowerShell.",
    "Created dynamic web apps with React, Angular, and Vue.js.",
    "Designed serverless architectures using AWS Lambda and API Gateway.",
    "Implemented CI/CD pipelines with Jenkins and GitHub Actions.",
    "Used Firebase for real-time databases and authentication.",
    "Developed GraphQL APIs with Apollo Server and Prisma.",
    "Managed NoSQL databases like MongoDB and Cassandra.",
    "Optimized PostgreSQL queries for high-traffic applications.",
    "Configured infrastructure as code with Terraform and Ansible.",
    "Set up monitoring with Prometheus, Grafana, and ELK stack.",
    "Deployed multi-cloud solutions on AWS, Azure, and GCP.",
    "Trained NLP models using spaCy, NLTK, and Hugging Face.",
    "Analyzed big data with PySpark and Pandas.",
    "Implemented computer vision solutions with OpenCV.",
    "Built recommendation systems using collaborative filtering.",
    "Deployed ML models with Flask, FastAPI, and ONNX runtime.",
    "Led Agile teams with Jira, Confluence, and daily standups.",
    "Practiced Test-Driven Development (TDD) with pytest and Jest.",
    "Used Docker Compose for local development environments.",
    "Managed version control with GitLab and Bitbucket.",
    "Applied SOLID principles and clean code architecture.",
    "Secured APIs using OAuth2, JWT, and rate limiting.",
    "Performed penetration testing with Burp Suite and Metasploit.",
    "Architected hybrid cloud solutions with Kubernetes clusters.",
    "Automated security scans with SonarQube and Snyk.",
    "Hardened Linux servers with SELinux and iptables."
    ];
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        print(f"Skills: {skills}")

if __name__ == "__main__":
    test_ner_model()
