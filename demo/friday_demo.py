"""Demo: run a simple FRIDAY query using the local backend.

Usage:
    python demo/friday_demo.py --mode very-low

This demo expects LOCAL_MODEL_PATH to be set in the environment or config.example.env.
"""

import os
import argparse
from src.friday.pipeline import FridayPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=os.environ.get('MODE', 'very-low'))
    args = parser.parse_args()

    # initialize pipeline
    model_path = os.environ.get('LOCAL_MODEL_PATH')
    fp = FridayPipeline(model_path=model_path, threads=int(os.environ.get('THREADS',4)), mode=args.mode)

    question = "Summarize the core idea of FRIDAY and list three optimization tips for low-RAM systems."
    print(f"Question: {question}\n---\nResponse:\n")
    try:
        for tok in fp.ask(question, max_tokens=200):
            print(tok, end='', flush=True)
    except Exception as e:
        print('\n\n[FRIDAY] Error: ', e)
        print('\nMake sure LOCAL_MODEL_PATH points to a quantized local model and that you installed gpt4all or llama-cpp-python.')

if __name__ == '__main__':
    main()
