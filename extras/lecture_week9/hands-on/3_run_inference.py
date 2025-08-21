# Filename: 3_run_inference.py
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Path to the final saved model (not a placeholder)
model_path = "./results"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create NER pipeline
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

# Test text
text = "A new CISA alert links the malware Guloader to the threat actor APT42."
df = pd.DataFrame(ner_pipeline(text))
print(df)
