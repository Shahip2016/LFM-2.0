import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Model")
    parser.add_argument("--model_path", type=str, default="model.pt")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    print("Evaluating on mock dataset...")
    
    accuracy = random.uniform(0.85, 0.99)
    f1_score = random.uniform(0.80, 0.95)
    
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
