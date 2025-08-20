from transformers import pipeline

# Load our fine-tuned model from the results directory
ner_pipeline = pipeline("ner", model="./results/checkpoint-1125")
text = "A new CISA alert links the malware Guloader to the threat actor APT42."
print(ner_pipeline(text))
