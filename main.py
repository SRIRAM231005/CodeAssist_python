import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from tqdm import tqdm


MODEL_NAME = "microsoft/codebert-base"
DEVICE = "cpu"
MAX_LEN = 256
BATCH_SIZE = 16  
TARGET_SIZE = 20_000

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

dataset = load_dataset("Nan-Do/code-search-net-python", split="train")
dataset = dataset.shuffle(seed=42)


def length_filter(example):
    code_len = len(example["code_tokens"])
    doc_len = len(example["docstring_tokens"])
    return (100 <= code_len <= 400) and (10 <= doc_len <= 100)

filtered_dataset = dataset.filter(length_filter)
filtered_dataset = filtered_dataset.select(range(TARGET_SIZE))

print(f"Using {len(filtered_dataset)} samples")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (B, T, 768)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts  # (B, 768)

all_embeddings = []
all_metadata = []

for start in tqdm(range(0, len(filtered_dataset), BATCH_SIZE)):
    end = start + BATCH_SIZE
    batch = filtered_dataset[start:end]

    code_texts = batch["code"]

    inputs = tokenizer(
        code_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pooled = mean_pooling(outputs, inputs["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    pooled_np = pooled.cpu().numpy()

    for i, vec in enumerate(pooled_np):
        all_embeddings.append(vec)

        meta = {
            "repo": batch["repo"][i],
            "path": batch["path"][i],
            "func_name": batch["func_name"][i],
            "docstring": batch["docstring"][i],
            "summary": batch["summary"][i],
        }

        all_metadata.append(meta)

all_embeddings = np.array(all_embeddings)  # (20000, 768)

print("Embeddings shape:", all_embeddings.shape)
print("Metadata count:", len(all_metadata))
print("Sample metadata:", all_metadata[0])

np.save("data/codebert_embeddings.npy", all_embeddings)
np.save("data/metadata.npy", all_metadata)
