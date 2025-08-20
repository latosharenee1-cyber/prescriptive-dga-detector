# Filename: 1_prepare_data.py
import json

# Define the dataset with custom labels
# 0: Other, 1: B-THREAT (Beginning of Threat), 2: I-THREAT (Inside Threat)
data = [
  {
    "tokens": ["The", "malware", "Guloader", "was", "used", "by", "APT42", "."],
    "ner_tags": [0, 0, 1, 0, 0, 0, 1, 0] 
  },
  {
    "tokens": ["We", "attribute", "this", "activity", "to", "Fancy", "Bear", "."],
    "ner_tags": [0, 0, 0, 0, 0, 1, 2, 0]
  },
  {
    "tokens": ["The", "attack", "leveraged", "the", "Cobalt", "Strike", "framework", "."],
    "ner_tags": [0, 0, 0, 0, 1, 2, 2, 0]
  }
]

# Save the data to a file
with open('threat_reports.json', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print("threat_reports.json created successfully.")

