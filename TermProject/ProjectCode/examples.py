import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load lexical feature data
df = pd.read_csv("text_features_lexical.csv")

# Infer label from clip_id path
df["label"] = df.clip_id.str.contains("nodementia", case=False).map({True: 0, False: 1})


# ✅ Use label instead of clip_id to split
df_dementia = df[df.label == 1]
df_control = df[df.label == 0]

# Select top dementia example by lexical entropy
high_entropy = df_dementia.sort_values("lex_entropy", ascending=False).iloc[0]
high_entropy_clip = high_entropy["clip_id"]

# Select top dementia example by semantic drift
high_drift = df_dementia.sort_values("sem_drift", ascending=False).iloc[0]
high_drift_clip = high_drift["clip_id"]

# Select low entropy/drift control
if not df_control.empty:
    low_drift_entropy = df_control.sort_values(["sem_drift", "lex_entropy"]).iloc[0]
    control_clip = low_drift_entropy["clip_id"]
else:
    print("Warning: No control (label=0) clips found. Using fallback.")
    low_drift_entropy = df.sort_values(["sem_drift", "lex_entropy"]).iloc[0]
    control_clip = low_drift_entropy["clip_id"]

# Function to load transcript
def load_transcript(clip_id, transcript_dir="transcripts"):
    clip_name = Path(clip_id).stem + ".txt"
    transcript_path = Path(transcript_dir) / clip_name
    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "[Transcript not found]"

# Load transcript snippets
samples = {
    "High Lexical Entropy (Dementia)": (high_entropy_clip, load_transcript(high_entropy_clip)),
    "High Semantic Drift (Dementia)": (high_drift_clip, load_transcript(high_drift_clip)),
    "Control (Low Drift/Entropy)": (control_clip, load_transcript(control_clip)),
}

# Create dataframe with preview
snippet_data = []
for label, (clip_id, text) in samples.items():
    snippet = " ".join(text.split()[:40]) + "..."  # preview first 40 words
    snippet_data.append([label, clip_id, snippet])

df_snippets = pd.DataFrame(snippet_data, columns=["Category", "Clip ID", "Transcript Snippet"])

# Display table figure
def display_table(dataframe):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    table = ax.table(cellText=dataframe.values,
                     colLabels=dataframe.columns,
                     cellLoc='left',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    plt.tight_layout()
    plt.savefig("snippet_examples.png", dpi=300)
    plt.show()

display_table(df_snippets)
