from fastapi import FastAPI, Body, UploadFile, File
from datetime import datetime, timedelta
from typing import List
import requests
import fitz  # PyMuPDF
import json

app = FastAPI()

OLLAMA_URL = "http://192.168.31.124:11434/api"
WEAVIATE_OBJECTS_URL = "http://localhost:8080/v1/objects"
WEAVIATE_GRAPHQL_URL = "http://localhost:8080/v1/graphql"

# Helper: 修正時間格式
def get_rfc3339_time():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# 1. 插入記憶
@app.post("/remember")
def remember(data: dict = Body(...)):
    text = data["text"]
    vector = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": "nomic-embed-text", "prompt": text}).json()["embedding"]

    obj = {
        "class": "Knowledge",
        "properties": {
            "text": text,
            "type": data.get("type", "fact"),
            "tag": data.get("tag", []),
            "domain": data.get("domain", "chat"),
            "source_doc": data.get("source_doc", "chat"),
            "created_at": data.get("created_at", get_rfc3339_time()),
            "related_event": data.get("related_event", ""),
            "user": data.get("user", "default")
        },
        "vector": vector
    }
    return requests.post(WEAVIATE_OBJECTS_URL, json=obj).json()

# 2. 查詢記憶
@app.post("/recall")
def recall(data: dict = Body(...)):
    text = data["text"]
    vector = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": "nomic-embed-text", "prompt": text}).json()["embedding"]

    query = {
        "query": f"""
        {{
          Get {{
            Knowledge(
              nearVector: {{ vector: {vector} }},
              limit: 5
            ) {{
              text type tag domain user source_doc created_at
              _additional {{ id certainty }}
            }}
          }}
        }}"""
    }
    return requests.post(WEAVIATE_GRAPHQL_URL, json=query).json()

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
@app.post("/upload_pdf")
def upload_pdf(file: UploadFile):
    text_blocks = []
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    for page in doc:
        text_blocks += [p.strip() for p in page.get_text().split("\n") if len(p.strip()) >= 30]

    success, fail = 0, 0
    for block in text_blocks:
        emb_res = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": "nomic-embed-text", "prompt": block})
        if emb_res.status_code != 200:
            fail += 1
            continue
        vector = emb_res.json()["embedding"]
        obj = {
            "class": "Knowledge",
            "properties": {
                "text": block,
                "type": "doc",
                "tag": ["pdf"],
                "domain": "FlexRIC",
                "source_doc": file.filename,
                "created_at": get_rfc3339_time(),
                "user": "system"
            },
            "vector": vector
        }
        res = requests.post(WEAVIATE_OBJECTS_URL, json=obj)
        success += 1 if res.status_code == 200 else 0
    return {"filename": file.filename, "total": len(text_blocks), "success": success, "fail": fail}

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
    query = {
        "query": f"""
        {{
          Get {{
            Knowledge(where: {{path: [\"created_at\"], operator: LessThan, valueDate: \"{cutoff}\"}}) {{
              _additional {{ id }}
            }}
          }}
        }}"""
    }
    resp = requests.post(WEAVIATE_GRAPHQL_URL, json=query).json()
    knowledge_data = resp.get("data", {}).get("Get", {}).get("Knowledge", [])
    if not knowledge_data:
        return {"deleted_count": 0, "cutoff": cutoff, "note": "No expired memory found."}

    deleted = 0
    for item in knowledge_data:
        id_ = item["_additional"]["id"]
        requests.delete(f"{WEAVIATE_OBJECTS_URL}/{id_}")
        deleted += 1
    return {"deleted_count": deleted, "cutoff": cutoff}
