import tarfile
import json
import random
import os
from pathlib import Path

def main():
    tar_path = "triviaqa-rc.tar.gz"
    evidence_dir = Path("data/triviaqa/evidence")
    eval_json_path = Path("data/eval_dataset.json")
    
    num_samples = 100
    
    # Create the evidence directory if it doesn't exist
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {tar_path} (this might take a minute, it's 2.6GB!)...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            print("Scanning archive for the Wikipedia QA JSON file...")
            
            # 1. Find the QA JSON
            qa_file_member = None
            for member in tar.getmembers():
                if "qa/" in member.name and "wikipedia-train.json" in member.name:
                    qa_file_member = member
                    break
            
            # Fallback if wikipedia-train isn't found
            if not qa_file_member:
                qa_file_member = next((m for m in tar.getmembers() if "qa/" in m.name and m.name.endswith(".json")), None)
                
            if not qa_file_member:
                print("Could not find a QA JSON file in the archive.")
                return
                
            print(f"Loading {qa_file_member.name} into memory...")
            f = tar.extractfile(qa_file_member)
            if not f:
                print("Failed to extract JSON file.")
                return
                
            data = json.load(f)
            
            items = data.get("Data", [])
            print(f"Found {len(items)} questions in dataset. Randomly sampling {num_samples}...")
            
            if len(items) > num_samples:
                sampled_items = random.sample(items, num_samples)
            else:
                sampled_items = items
                
            eval_dataset = []
            evidence_filenames = set()
            
            for item in sampled_items:
                question = item.get("Question")
                answer = item.get("Answer", {}).get("Value", "")
                eval_dataset.append({
                    "question": question,
                    "ground_truth": answer
                })
                
                # Identify required evidence files
                for ep in item.get("EntityPages", []):
                    fname = ep.get("Filename")
                    if fname:
                        evidence_filenames.add(fname)
                        
            print(f"The 100 questions require {len(evidence_filenames)} unique Wikipedia evidence documents.")
            print("Scanning archive to extract only the required evidence documents...")
            
            # 2. Extract only the required evidence files
            extracted_count = 0
            for member in tar.getmembers():
                if extracted_count >= len(evidence_filenames):
                    break
                
                basename = os.path.basename(member.name)
                if basename in evidence_filenames:
                    tar.extract(member, path=evidence_dir)
                    extracted_count += 1
                    if extracted_count % 10 == 0:
                        print(f"Extracted {extracted_count}/{len(evidence_filenames)} files...")
            
            # 3. Save the evaluation dataset
            print(f"\nWriting {len(eval_dataset)} evaluation Q&A pairs to {eval_json_path}...")
            with open(eval_json_path, "w", encoding="utf-8") as out_f:
                json.dump(eval_dataset, out_f, indent=2)
                
            print("\n✅ Preparation Complete!")
            print("--------------------------------------------------")
            print("Next steps:")
            print("1. Ingest the new evidence into Qdrant (this will clear old data):")
            print("   uv run python scripts/ingest_docs.py --source data/triviaqa/evidence --clear")
            print("\n2. Run the RAGAS evaluation on the 100 sampled questions:")
            print("   uv run python scripts/evaluate.py")
            
    except Exception as e:
        print(f"Error processing tar file: {e}")

if __name__ == "__main__":
    main()
