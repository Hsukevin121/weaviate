from fastapi import FastAPI, Body, UploadFile, File
from datetime import datetime, timedelta
from typing import List
import requests
import fitz  # PyMuPDF
import json
import logging
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uuid
from docx import Document
import tiktoken

# 設定 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 rerank 模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
model.eval()

# 封裝 rerank 函數
def rerank(query: str, documents: list[str], top_k: int = 5, threshold: float = 0.8):
    pairs = [(query, doc) for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).tolist()

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [{"text": doc, "score": score} for doc, score in ranked if score >= threshold][:top_k]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 http://192.168.31.132:5173 更安全
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OLLAMA_URL = "http://10.8.0.26:11434/api"
WEAVIATE_OBJECTS_URL = "http://localhost:8080/v1/objects"
WEAVIATE_GRAPHQL_URL = "http://localhost:8080/v1/graphql"

def get_rfc3339_time():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

@app.post("/remember")
def remember(data: dict = Body(...)):
    try:
        text = data.get("text")
        if not text:
            return {"error": "Missing 'text' field"}

        # === 1. 呼叫 embedding 模型 ===
        emb_response = requests.post(
            f"{OLLAMA_URL}/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        emb_json = emb_response.json()
        vector = emb_json.get("embedding")

        if not vector or not isinstance(vector, list):
            logger.error(f"[embedding error] response: {emb_json}")
            return {"error": "embedding failed", "response": emb_json}

        # === 2. 準備 Weaviate 物件 ===
        obj = {
            "id": str(uuid.uuid4()),
            "class": "Knowledge",
            "properties": {
                "text": text,
                "type": data.get("type", "fact"),
                "tag": data.get("tag", []),
                "domain": data.get("domain", "chat"),
                "source_doc": data.get("source_doc", "chat"),
                "created_at": data.get("created_at", get_rfc3339_time()),
                "related_event": data.get("related_event", ""),
                "user": data.get("user", "default"),
                "certainty": data.get("certainty", 1.0)
            },
            "vector": vector
        }

        # === 3. 寫入 Weaviate ===
        headers = {"Content-Type": "application/json"}
        res = requests.post(WEAVIATE_OBJECTS_URL, json=obj, headers=headers)

        if res.status_code >= 300:
            logger.error(f"[weaviate insert error] {res.status_code} {res.text}")
            return {"error": "weaviate insertion failed", "response": res.text}

        return {"message": "success", "id": obj["id"], "weaviate_response": res.json()}

    except Exception as e:
        logger.exception("[remember] unexpected error")
        return {"error": "unexpected failure", "exception": str(e)}

@app.post("/recall")
def recall(data: dict = Body(...)):
    try:
        query = data["text"]
        emb = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": "nomic-embed-text", "prompt": query}).json()
        vector = emb["embedding"]

        query_body = {
            "query": f"""
            {{
              Get {{
                Knowledge(nearVector: {{ vector: {vector} }}, limit: 20) {{
                  text source_doc created_at _additional {{ id certainty }}
                }}
              }}
            }}"""
        }
        res = requests.post(WEAVIATE_GRAPHQL_URL, json=query_body).json()
        candidates = res.get("data", {}).get("Get", {}).get("Knowledge", [])
        docs = [c["text"] for c in candidates]

        top_docs = rerank(query, docs, top_k=5, threshold=0.8)

        return {
            "query": query,
            "selected_docs": top_docs,
            "raw_candidates": docs[:10]
        }
    except Exception as e:
        logger.error(f"[recall] error: {e}")
        return {"error": "recall failed", "exception": str(e)}

# 3. 手動標記
@app.post("/label")
def label(data: dict = Body(...)):
    object_id = data["id"]
    patch_data = {"properties": data["properties"]}
    res = requests.patch(f"{WEAVIATE_OBJECTS_URL}/Knowledge/{object_id}", json=patch_data)
    if res.status_code == 204:
        return {"status": "success", "id": object_id}
    try:
        return res.json()
    except Exception:
        return {"error": "patch failed", "status_code": res.status_code, "raw": res.text}

# 4. 建議 label（LLM 判斷）
@app.post("/suggest_label")
def suggest_label(data: dict = Body(...)):
    text = data["text"]
    prompt = (
        "請你根據以下內容，輸出分類 type 和 tag。\n"
        "- 回傳格式：{\"type\": \"fact\", \"tag\": [\"flexric\"]}\n"
        "- type 請從 fact, policy, task, doc 選一\n"
        "- tag 請用陣列方式回傳，並且內容為英文詞\n\n"
        f"內容如下：\n{text}"
    )
    res = requests.post(f"{OLLAMA_URL}/generate", json={"model": "llama3.1:latest", "prompt": prompt, "stream": False})
    try:
        content = res.json()["response"]
        json_str = content[content.find("{"):content.rfind("}")+1]
        result = json.loads(json_str)
        if isinstance(result.get("tag"), str):
            result["tag"] = [result["tag"]]
        return result
    except Exception as e:
        return {"error": "解析 LLM 回應失敗", "raw": res.text, "exception": str(e)}

# 5. 刪除記憶
@app.delete("/delete/{object_id}")
def delete(object_id: str):
    res = requests.delete(f"{WEAVIATE_OBJECTS_URL}/{object_id}")
    return {"status": "success", "id": object_id} if res.status_code == 204 else {"status": "fail", "code": res.status_code, "reason": res.text}

# 6. 條件查詢
@app.post("/query_by_filter")
def query_by_filter(data: dict = Body(...)):
    where_clause = []
    for key in ["type", "domain"]:
        if key in data:
            where_clause.append(f'{{path: ["{key}"], operator: Equal, valueText: "{data[key]}"}}')
    if "tag" in data:
        tags = json.dumps(data["tag"])
        where_clause.append(f'{{path: ["tag"], operator: ContainsAny, valueText: {tags}}}')

    where_str = f'where: {{operands: [{", ".join(where_clause)}], operator: And}}' if where_clause else ""

    query = {
        "query": f"""
        {{
          Get {{
            Knowledge({where_str}) {{
              text type tag domain user created_at _additional {{ id }}
            }}
          }}
        }}"""
    }
    return requests.post(WEAVIATE_GRAPHQL_URL, json=query).json()

# 7. PDF 拆段上傳
def split_into_chunks(text: str, max_tokens=200, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        chunk = tokens[start:start + max_tokens]
        decoded_chunk = enc.decode(chunk)
        chunks.append(decoded_chunk)
        start += max_tokens - overlap

    return chunks

@app.post("/upload_file")
def upload_file(file: UploadFile):
    filename = file.filename.lower()
    ext = filename.split(".")[-1]
    full_text = ""

    # === Step 1: 擷取全文文字 ===
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        for page in doc:
            full_text += page.get_text() + "\n"

    elif filename.endswith(".docx"):
        doc = Document(file.file)
        full_text = "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".txt"):
        content = file.file.read().decode("utf-8")
        full_text = content

    else:
        return {"error": "Unsupported file type"}

    # === Step 2: 切成語意片段 ===
    text_blocks = split_into_chunks(full_text)

    success, fail = 0, 0
    for i, block in enumerate(text_blocks):
        try:
            emb_res = requests.post(f"{OLLAMA_URL}/embeddings", json={
                "model": "nomic-embed-text",
                "prompt": block
            })

            if emb_res.status_code != 200:
                print(f"❌ Failed to embed chunk {i}")
                fail += 1
                continue

            vector = emb_res.json()["embedding"]
            obj = {
                "class": "Knowledge",
                "properties": {
                    "text": block,
                    "type": "doc",
                    "tag": [ext],
                    "domain": "O-RAN",
                    "source_doc": file.filename,
                    "chunk_index": i,
                    "created_at": get_rfc3339_time(),
                    "user": "system"
                },
                "vector": vector
            }

            res = requests.post(WEAVIATE_OBJECTS_URL, json=obj)
            if res.status_code == 200:
                success += 1
            else:
                print(f"❌ Weaviate insert failed for chunk {i}: {res.text}")
                fail += 1

        except Exception as e:
            print(f"⚠️ Chunk {i} failed: {e}")
            fail += 1

    return {
        "filename": file.filename,
        "total_chunks": len(text_blocks),
        "success": success,
        "fail": fail
    }

# 8. 查詢使用者記憶
@app.get("/history/{user}")
def user_history(user: str):
    query = {
        "query": f"""
        {{
          Get {{
            Knowledge(where: {{path: [\"user\"], operator: Equal, valueText: \"{user}\"}}) {{
              text created_at type tag domain
            }}
          }}
        }}"""
    }
    return requests.post(WEAVIATE_GRAPHQL_URL, json=query).json()

# 9. 清理過期記憶
@app.delete("/clean_old")
def clean_old(days: int = 30):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds") + "Z"
    deleted = 0
    limit = 100
    after = None

    while True:
        # 建立 GraphQL 查詢字串（加上 after 分頁游標）
        after_clause = f', after: "{after}"' if after else ""
        query = {
            "query": f"""{{
                Get {{
                    Knowledge(
                        where: {{
                            path: ["created_at"],
                            operator: LessThan,
                            valueDate: "{cutoff}"
                        }},
                        limit: {limit}{after_clause}
                    ) {{
                        _additional {{ id }}
                    }}
                }}
            }}"""
        }

        # 執行查詢
        response = requests.post(WEAVIATE_GRAPHQL_URL, json=query).json()
        objects = response.get("data", {}).get("Get", {}).get("Knowledge", [])

        if not objects:
            break  # 沒有更多資料了

        for obj in objects:
            obj_id = obj["_additional"]["id"]
            requests.delete(f"{WEAVIATE_OBJECTS_URL}/Knowledge/{obj_id}")
            deleted += 1

        # 分頁：以最後一筆資料為 after 游標
        after = objects[-1]["_additional"]["id"]

    return {
        "deleted_count": deleted,
        "cutoff": cutoff,
        "note": "All expired memory deleted completely."
    }
