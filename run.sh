#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Choose a mode:"
echo "  ./run.sh ingest                          -> process PDFs and update the database"
echo "  ./run.sh query \"your question\" [top_k]   -> ask a question"
echo ""

MODE=$1

if [ "$MODE" == "ingest" ]; then
    python -m src.ingest
elif [ "$MODE" == "query" ]; then
    QUESTION=$2
    TOP_K=${3:-3}
    python -m src.query "$QUESTION" "$TOP_K"
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run.sh ingest | ./run.sh query \"question\" [top_k]"
    exit 1
fi