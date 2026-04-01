import pandas as pd

# Load dataset
df = pd.read_csv("faq_dataset.csv")

# Clean text
df["text"] = df["text"].astype(str).str.replace('"', '', regex=False).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

# Sort
df = df.sort_values(by=["intent", "text"])

# Add spacing
rows = []
current_intent = None

for _, row in df.iterrows():
    if current_intent is not None and row["intent"] != current_intent:
        rows.append({"text": "", "intent": ""})  # blank line

    rows.append({
        "text": row["text"],
        "intent": row["intent"]
    })

    current_intent = row["intent"]

# Convert to DataFrame
df_spaced = pd.DataFrame(rows, columns=["text", "intent"])

# Save
df_spaced.to_csv("faq_dataset_sorted.csv", index=False)

print("✅ Sorted dataset created!")