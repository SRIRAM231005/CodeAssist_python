import os
import json
import ast
import requests
import streamlit as st
import torch
import numpy as np
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

from transformers import RobertaTokenizer, RobertaModel

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DEVICE = "cpu"
CODEBERT_MODEL = "microsoft/codebert-base"
CHROMA_PATH = "../chroma_store"
COLLECTION_NAME = "codebert_python"
TOP_K = 5

tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
codebert = RobertaModel.from_pretrained(CODEBERT_MODEL).to(DEVICE)
codebert.eval()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

gemini = genai.GenerativeModel("gemini-2.5-flash")

# Fetching Github Files and extracting code

def fetch_github_file(repo_url, file_path, branch="main"):
    clean_repo = repo_url.rstrip("/").replace("https://github.com/", "")
    raw_url = f"https://raw.githubusercontent.com/{clean_repo}/{branch}/{file_path}"
    response = requests.get(raw_url)

    if response.status_code == 200:
        return response.text
    elif response.status_code == 404:
        raise ValueError("File not found. Check file path or branch.")
    else:
        raise RuntimeError(f"GitHub request failed: {response.status_code}")

def extract_functions_from_code(code: str):
    tree = ast.parse(code)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "code": ast.get_source_segment(code, node)
            })

        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions.append({
                        "name": f"{node.name}.{item.name}",
                        "code": ast.get_source_segment(code, item)
                    })

    return functions

# Getting structural intent in json format from User Query

def parse_user_query(user_query: str):
    prompt = f"""
Return ONLY valid JSON.

Fields:
- intent: one of [explain, bug_explanation, usage, optimization, refactor, security, general_question]
- function_name_candidates: list of function or method names if mentioned
- raw_query

User query:
"{user_query}"
"""
    response = gemini.generate_content(prompt)
    text = response.text.strip()
    text = text[text.find("{"):text.rfind("}")+1]
    return json.loads(text)


# Filtering Functions by name

def filter_functions(functions, name_candidates):
    if not name_candidates:
        return functions

    name_candidates = [n.lower() for n in name_candidates]

    filtered = [
        fn for fn in functions
        if fn["name"].lower() in name_candidates
    ]

    return filtered if filtered else functions


# Embedding User Query using CodeBERT

def mean_pool(outputs, attention_mask):
    token_embeds = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    summed = torch.sum(token_embeds * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_query(text: str):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = codebert(**inputs)

    return mean_pool(outputs, inputs["attention_mask"]).cpu().numpy()


# Chroma Similarity Search

def retrieve_similar(query_embedding):
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )


# Generating Final Explanation

def generate_explanation(
    local_functions,
    retrieved_context,
    intent,
    raw_query
):
    prompt = f"""
You are a senior software engineer.

User intent: {intent}
User query: {raw_query}

=====================
LOCAL CODE (from repo)
=====================
{json.dumps(local_functions, indent=2)}

=====================
RETRIEVED REFERENCES
=====================
{json.dumps(retrieved_context, indent=2)}

Give a clear, accurate, and structured explanation.
"""
    response = gemini.generate_content(prompt)
    return response.text


# Streamlit UI

st.set_page_config(page_title="CodeAssist", layout="wide")
st.header(":material/code: CodeAssist")
st.caption("Intelligent Code Explainer")

with st.form("input_form"):
    repo_url = st.text_input("GitHub Repository URL")
    file_path = st.text_input("File Path (inside repo)")
    user_query = st.text_input("Your Question")
    submitted = st.form_submit_button("Analyze")

if submitted:
    with st.spinner("Fetching source code..."):
        code_text = fetch_github_file(repo_url, file_path)

    with st.spinner("Extracting functions..."):
        functions = extract_functions_from_code(code_text)

    with st.spinner("Understanding your query..."):
        parsed_query = parse_user_query(user_query)

    with st.spinner("Filtering relevant functions..."):
        filtered_functions = filter_functions(
            functions,
            parsed_query["function_name_candidates"]
        )

    with st.spinner("Embedding user query..."):
        query_embedding = embed_query(user_query)

    with st.spinner("Retrieving similar code knowledge..."):
        results = retrieve_similar(query_embedding)

    retrieved_context = []
    for docs, metas in zip(results["documents"], results["metadatas"]):
        for d, m in zip(docs, metas):
            retrieved_context.append({
                "summary": d,
                "metadata": m
            })

    with st.spinner("Generating explanation..."):
        explanation = generate_explanation(
            filtered_functions,
            retrieved_context,
            parsed_query["intent"],
            parsed_query["raw_query"]
        )

    st.subheader("📌 Explanation")
    st.markdown(explanation)
