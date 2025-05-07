import re
import json
import logging
import os
from datetime import datetime
import requests
import fitz  # PyMuPDF
import docx
from fastapi import FastAPI, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# === Logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://192.168.31.124:11434/api"
WEAVIATE_OBJECTS_URL = "http://localhost:8080/v1/objects"

# === 工具函式 ===
def get_rfc3339_time():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def error_response(msg: str, e: Exception = None, detail: dict = None):
    return {
        "status": "error",
        "message": msg,
        "exception": str(e) if e else None,
        "detail": detail or {}
    }

def get_embedding(text: str, model_name: str = "nomic-embed-text"):
    try:
        res = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": model_name, "prompt": text}, timeout=10)
        res.raise_for_status()
        return res.json()["embedding"]
    except Exception as e:
        logger.error(f"[embedding error] {e}")
        raise

def extract_json_from_llm_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("無法從 LLM 回應中解析 JSON 區塊")
    return json.loads(match.group())

def suggest_label(text: str):
    prompt = (
        "請你根據以下內容，輸出分類 type 和 tag。\n"
        "- 回傳格式：{\"type\": \"fact\", \"tag\": [\"flexric\"]}\n"
        "- type 請從 fact, policy, task, doc 選一\n"
        "- tag 請用陣列方式回傳，並且內容為英文詞\n\n"
        f"內容如下：\n{text}"
    )
    try:
        res = requests.post(
            f"{OLLAMA_URL}/generate",
            json={"model": "myaniu/qwen2.5-1m:7b", "prompt": prompt, "stream": False},
            timeout=20
        )
        res.raise_for_status()
        content = res.json().get("response")
        if not content:
            raise ValueError("缺少 'response' 欄位")
        parsed = extract_json_from_llm_response(content)
        if isinstance(parsed.get("tag"), str):
            parsed["tag"] = [parsed["tag"]]
        return parsed
    except Exception as e:
        logger.warning(f"[suggest_label fallback] {e}")
        return {"type": "doc", "tag": []}  # fallback

def insert_to_weaviate(text: str, vector: list, metadata: dict):
    obj = {
        "class": "Knowledge",
        "properties": {
            "text": text,
            **metadata
        },
        "vector": vector
    }
    return requests.post(WEAVIATE_OBJECTS_URL, json=obj, timeout=10)

def process_paragraph_block(text_block: str, filename: str, domain: str):
    try:
        embedding = get_embedding(text_block)
        label_info = suggest_label(text_block)

        metadata = {
            "type": label_info.get("type", "doc"),
            "tag": label_info.get("tag", [os.path.splitext(filename)[1].replace('.', '')]),
            "domain": domain,
            "source_doc": filename,
            "created_at": get_rfc3339_time(),
            "user": "system"
        }

        response = insert_to_weaviate(text_block, embedding, metadata)
        return response.status_code == 200

    except Exception as e:
        logger.error(f"[process block error] {e}")
        return False

# === FastAPI 應用 ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_file")
def upload_file(
    file: UploadFile = File(...),
    domain: str = Form("General"),
    user_tag: str = Form("")
):
    try:
        text_blocks = []
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            doc = fitz.open(stream=file.file.read(), filetype="pdf")
            for page in doc:
                text_blocks += [p.strip() for p in page.get_text().split("\n") if len(p.strip()) >= 30]
        elif filename.endswith(".txt"):
            content = file.file.read().decode("utf-8", errors="ignore")
            text_blocks = [line.strip() for line in content.splitlines() if len(line.strip()) >= 30]
        elif filename.endswith(".docx"):
            doc = docx.Document(file.file)
            text_blocks = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) >= 30]
        else:
            return error_response("不支援的檔案格式")

        success, fail = 0, 0
        for block in text_blocks:
            ok = process_paragraph_block(block, filename=file.filename, domain=domain, user_tag=user_tag)
            if ok:
                success += 1
            else:
                fail += 1

        return {"filename": file.filename, "total": len(text_blocks), "success": success, "fail": fail}
    except Exception as e:
        return error_response("file upload failed", e)

@app.post("/recall")
def recall(data: dict = Body(...)):
    try:
        query = data["text"]
        vector = get_embedding(query)

        graphql_query = {
            "query": f"""
            {{
              Get {{
                Knowledge(nearVector: {{ vector: {vector} }}, limit: 20) {{
                  text source_doc created_at _additional {{ id certainty }}
                }}
              }}
            }}"""
        }
        res = requests.post(WEAVIATE_GRAPHQL_URL, json=graphql_query).json()
        candidates = res.get("data", {}).get("Get", {}).get("Knowledge", [])
        docs = [c["text"] for c in candidates]

        top_docs = rerank(query, docs)

        return {
            "query": query,
            "results": top_docs,
            "raw": candidates[:10]
        }
    except Exception as e:
        return error_response("recall failed", e)