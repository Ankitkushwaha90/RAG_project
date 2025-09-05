@echo off
REM Activate your conda environment (replace rag_demo with your env name)
call conda activate rag_demo

REM Build the index from sample_docs folder
python rag.py --build-corpus sample_docs

REM Ask a sample question
python rag.py --ask "What is Retrieval-Augmented Generation?"

pause
