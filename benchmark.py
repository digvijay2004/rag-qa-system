import json
import time
import mlflow
from rag import answer_question
from ingest import ingest_pdf

# 25 Q&A pairs based on your resume
QA_PAIRS = [
    {
        "question": "What is the candidate's highest level of education?",
        "expected": "Master's degree in Computer Science from Western Michigan University"
    },
    {
        "question": "What programming languages does the candidate know?",
        "expected": "Python, SQL, JavaScript, R, C#"
    },
    {
        "question": "What AI frameworks does the candidate use?",
        "expected": "LangChain, CrewAI, RAG Pipelines, Hugging Face Transformers"
    },
    {
        "question": "What is the candidate's most recent work experience?",
        "expected": "Software Engineer at Bright Mind Enrichment since February 2026"
    },
    {
        "question": "What cloud platforms does the candidate have experience with?",
        "expected": "AWS, Azure, GCP"
    },
    {
        "question": "What is the candidate's GPA or graduation year?",
        "expected": "Graduated December 2025"
    },
    {
        "question": "What databases does the candidate work with?",
        "expected": "PostgreSQL, MySQL, FAISS"
    },
    {
        "question": "What is the candidate's email address?",
        "expected": "d.bhgat20@gmail.com"
    },
    {
        "question": "What MLOps tools does the candidate use?",
        "expected": "MLflow, Weights and Biases, Docker, CI/CD"
    },
    {
        "question": "What certifications does the candidate have?",
        "expected": "DeepLearning.AI LangChain certification and Microsoft Generative AI certification"
    },
    {
        "question": "What was the candidate's role at Yash Infotech?",
        "expected": "Associate Software Engineer in Backend and Data Engineering"
    },
    {
        "question": "How many years of experience does the candidate have?",
        "expected": "Over 2 years of professional software engineering experience"
    },
    {
        "question": "What deep learning frameworks does the candidate know?",
        "expected": "PyTorch and TensorFlow"
    },
    {
        "question": "What is the candidate's GitHub username?",
        "expected": "digvijay2004"
    },
    {
        "question": "What was the candidate's internship role?",
        "expected": "Software Engineer Intern at Yash Infotech"
    },
    {
        "question": "What RAG project did the candidate build?",
        "expected": "RAG Powered Q&A System using LangChain FAISS Hugging Face and Groq"
    },
    {
        "question": "What is the candidate's phone number?",
        "expected": "+1 269 364 0990"
    },
    {
        "question": "What university did the candidate complete their bachelor's degree at?",
        "expected": "Savitribai Phule Pune University in India"
    },
    {
        "question": "What machine learning concepts does the candidate know?",
        "expected": "NLP Computer Vision Deep Learning Reinforcement Learning"
    },
    {
        "question": "What containerization tools does the candidate use?",
        "expected": "Docker"
    },
    {
        "question": "What is the candidate's GAN project about?",
        "expected": "Car image generation using Generative Adversarial Networks with PyTorch"
    },
    {
        "question": "What accuracy did the candidate achieve in the crash prediction project?",
        "expected": "87 percent accuracy"
    },
    {
        "question": "What LLM tool did the candidate use at Bright Mind?",
        "expected": "LangChain and OpenAI API for report summarization"
    },
    {
        "question": "What is the candidate's LinkedIn profile?",
        "expected": "linkedin.com/in/digvijay-bhagat"
    },
    {
        "question": "What agile methodology does the candidate follow?",
        "expected": "Agile Scrum"
    }
]

def compute_rouge_score(predicted: str, expected: str) -> float:
    predicted_words = set(predicted.lower().split())
    expected_words = set(expected.lower().split())
    if not expected_words:
        return 0.0
    overlap = predicted_words.intersection(expected_words)
    precision = len(overlap) / len(predicted_words) if predicted_words else 0
    recall = len(overlap) / len(expected_words) if expected_words else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)

def run_benchmark(pdf_path: str):
    print("Ingesting PDF...")
    ingest_pdf(pdf_path)
    print("PDF ingested successfully!")
    print(f"Running benchmark on {len(QA_PAIRS)} questions...")

    results = []
    rouge_scores = []

    mlflow.set_experiment("RAG Benchmark Evaluation")

    with mlflow.start_run(run_name="benchmark_run"):
        mlflow.log_param("num_questions", len(QA_PAIRS))
        mlflow.log_param("pdf_path", pdf_path)

        for i, qa in enumerate(QA_PAIRS):
            question = qa["question"]
            expected = qa["expected"]

            start_time = time.time()
            predicted = answer_question(question)
            latency = round(time.time() - start_time, 2)

            rouge = compute_rouge_score(predicted, expected)
            rouge_scores.append(rouge)

            result = {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "rouge_score": rouge,
                "latency_seconds": latency
            }
            results.append(result)

            print(f"Q{i+1}: ROUGE={rouge} | Latency={latency}s")
            print(f"  Question: {question}")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {predicted[:100]}...")
            print()

        avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4)
        avg_latency = round(
            sum(r["latency_seconds"] for r in results) / len(results), 2
        )

        mlflow.log_metric("avg_rouge_score", avg_rouge)
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        mlflow.log_metric("num_questions_evaluated", len(QA_PAIRS))

        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact("benchmark_results.json")

        print("=" * 50)
        print(f"BENCHMARK COMPLETE")
        print(f"Average ROUGE Score: {avg_rouge}")
        print(f"Average Latency: {avg_latency} seconds")
        print(f"Total Questions: {len(QA_PAIRS)}")
        print("=" * 50)

        return avg_rouge, avg_latency, results

if __name__ == "__main__":
    pdf_path = "Digvijay_Bhagat_Resume.pdf"
    run_benchmark(pdf_path)