# insert_data_flex.py

from datasets import load_dataset
import requests
import json
import time

OLLAMA_URL = "http://192.168.31.129:11434/api"
WEAVIATE_OBJECTS_URL = "http://localhost:8080/v1/objects"

# 自訂: 你的資料集清單
DATASETS = [
    {"name": "greenwich157/5G_Faults_Full", "tag": ["general_fault"], "fields": ("input", "output")},
    {"name": "greenwich157/5G-Faults-DPO", "tag": ["dpo_fault"], "fields": ("prompt", "chosen")},
    {"name": "greenwich157/telco_5G_general_faults", "tag": ["general_fault"], "fields": ("input", "output")},
    {"name": "greenwich157/telco-5G-core-faults", "tag": ["core_fault"], "fields": ("input", "output")},
    {"name": "greenwich157/telco-5g-data-faults", "tag": ["data_fault"], "fields": ("text", None)},
]

# 輔助: 向量化
def embed_text(text):
    res = requests.post(f"{OLLAMA_URL}/embeddings", json={"model": "nomic-embed-text", "prompt": text})
    return res.json().get("embedding", [])

# 輔助: 插入到 Weaviate
def insert_to_weaviate(text, tag, source_doc):
    vector = embed_text(text)
    obj = {
        "class": "Knowledge",
        "properties": {
            "text": text,
            "type": "doc",
            "tag": tag,
            "domain": "telecom",
            "source_doc": source_doc,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": "system"
        },
        "vector": vector
    }
    res = requests.post(WEAVIATE_OBJECTS_URL, json=obj)
    return res.status_code == 200

# 主程式
if __name__ == "__main__":
    for dataset_info in DATASETS:
        name = dataset_info["name"]
        tag = dataset_info["tag"]
        fields = dataset_info["fields"]
        print(f"\n➡️ 開始處理 {name}...")
        
        try:
            ds = load_dataset(name, split="train")
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            continue

        success, fail = 0, 0

        for item in ds:
            try:
                if fields[1]:
                    text = f"Q: {item[fields[0]]}\nA: {item[fields[1]]}"
                else:
                    text = item[fields[0]]

                if len(text.strip()) < 30:
                    continue  # 過短，跳過

                ok = insert_to_weaviate(text, tag, name)
                if ok:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"錯誤: {e}")
                fail += 1

        print(f"✅ {name} 上傳完成: 成功 {success} 筆，失敗 {fail} 筆")
