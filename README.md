# Alice RAG. Local Retrieval-Augmented Generation Demo

I made this project to create my first local Retrieval-Augmented Generation (RAG) system. I decided to use the public-domain literary corpus, Alice’s Adventures in Wonderland, This is because everyone is universally familiar with the general contents and it is also public domain. I also used Ollama to run the LLM locally without API dependencies. The guardrails and pipeline are part of my code. This project can be easily swapped to a hosted LLM.

I focused on getting the AI to give answers that are grounded and do not branch off into metaphorical language or hallucinate. One that refuses an answer when it does not have enough information.  


## Project Goals

- Demonstrate understanding of RAG architecture
- Prevent hallucinations through strict grounding
- Explicitly handle ambiguity and missing information
- Provide repeatable, logged evaluation
- Use public-domain data and fully local models


## System Overview

The system follows a standard RAG pipeline:

1. Knowledge Base

- Curated “knowledge cards” derived from the text
  
- Each card includes:
  
  - summary
    
  - ambiguity notes
    
  - guardrail instructions

2. Embedding & Indexing

- Knowledge cards are embedded using a sentence-transformer model
- Embeddings are stored locally in a JSON index

3. Retrieval

- User questions are embedded
  
- Top-K semantic matches are retrieved using cosine similarity
  
- Only the most relevant cards are used for answering

4. Answer Generation

- A lightweight local LLM (via Ollama) generates answers
  
- The model is constrained to:
  
  - use only retrieved context
    
  - cite card IDs
    
  - refuse when information is missing

5. Evaluation

  - A scripted evaluation runs a fixed question set
  - Results are logged to CSV for inspection


## Repository Structure
Alice-RAG/
- alice_kb.csv          # Curated knowledge cards
- build_index.py        # Builds vector index
- alice_index.json      # Generated embedding index
- ask.py                # Interactive Q&A interface
- eval_questions.txt    # Evaluation question set
- evaluate.py           # Evaluation runner
- eval_results.csv      # Evaluation output
- README.md


## How to Run
### 1. Build the index
python build_index.py


### 2. Ask questions interactively
python ask.py


### Example:
Question> How does Alice enter Wonderland?

## Run evaluation
python evaluate.py

- This generates eval_results.csv, which logs:
- retrieved card IDs
- similarity scores
- response latency
- refusal behavior
- final answers

  ## Guardrails & Safety



This system intentionally avoids:

- inventing plot details
- interpreting symbolism not stated in the text
- filling gaps with assumptions

If the knowledge cards do not explicitly contain the answer, the assistant responds:

  “I can’t answer that from the current knowledge cards.”

This behavior is intentional and evaluated.


## Evaluation Summary

- Retrieval consistently surfaces relevant context
- Answers remain grounded and cited
- Refusals occur appropriately
- Latency reflects local CPU inference and improves after warm-up

The evaluation script provides repeatable evidence of system behavior.






  
