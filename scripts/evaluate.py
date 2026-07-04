import json
import sys
import argparse
from pathlib import Path

# Add the project root to sys.path so we can import 'src'
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.config import get_settings
from src.services.rag import pipeline
from src.ml.llm import get_llm
from src.utils.logger import get_logger
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline using RAGAS")
    parser.add_argument("--dataset", type=str, default="data/eval_dataset.json", help="Path to evaluation dataset")
    args = parser.parse_args()

    dataset_path = Path(project_root) / args.dataset
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} evaluation questions.")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Run each question through the pipeline
    for idx, item in enumerate(data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        logger.info(f"Evaluating [{idx+1}/{len(data)}]: {question}")
        
        # We explicitly skip the agent for evaluation to focus on RAG precision
        result = pipeline.query(user_input=question, use_agent=False)
        
        # Extract the texts from citations to use as context
        retrieved_contexts = [cit.text for cit in result.citations]
        
        questions.append(question)
        answers.append(result.answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    # Format for RAGAS (HuggingFace Dataset)
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    logger.info("Starting RAGAS evaluation...")
    
    llm = get_llm()
    settings = get_settings()
    langchain_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
    # Configure metrics to use our LLM and embeddings
    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = langchain_embeddings

    # Run evaluate
    try:
        eval_result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=llm
        )
        
        logger.info("Evaluation complete!")
        print("\n--- RAGAS Evaluation Results ---")
        print(eval_result)
        
        # Save detailed results to CSV
        df = eval_result.to_pandas()
        csv_path = Path(project_root) / "data" / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")
        
        # Save summary to JSON
        summary_path = Path(project_root) / "data" / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            # Convert RagasResult to dict (it behaves like a dict for its scores)
            json.dump({k: float(v) for k, v in eval_result.items()}, f, indent=4)
        logger.info(f"Summary saved to {summary_path}")
        
    except Exception as e:
        logger.error("RAGAS Evaluation failed", extra={"error": str(e)}, exc_info=True)

if __name__ == "__main__":
    main()
