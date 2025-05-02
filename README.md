# weaviate
## 1. Vector DB installation
**Step 1 :** git clone resources
```=
git clone https://github.com/Hsukevin121/weaviate.git
cd ~/weaviate
```
**Step 2 :** run 
```=
docker compose up -d
```
Using the command to test : 
```=
curl http://localhost:8080/v1/meta
```
## 2. Design DB
**Step 1 :** create schema
```=
curl -X POST http://localhost:8080/v1/schema \
-H "Content-Type: application/json" \
-d '{
  "class": "Knowledge",
  "vectorizer": "none",
  "properties": [
    { "name": "text", "dataType": ["text"] },
    { "name": "type", "dataType": ["text"] },
    { "name": "tag", "dataType": ["text[]"] },
    { "name": "domain", "dataType": ["text"] },
    { "name": "source_doc", "dataType": ["text"] },
    { "name": "created_at", "dataType": ["date"] },
    { "name": "related_event", "dataType": ["text"] }
  ]
}'
```
![image](https://github.com/user-attachments/assets/e663ab30-b959-475b-9211-f0e199a94bee)

### 2-1. Architecture
**`Knowledge` class :** 
| 欄位名稱      | 類型   | 說明                                                 |
| ------------- | ------ | ---------------------------------------------------- |
| text          | text   | 向量內容的主要來源。所有查詢都會比對這欄位的語意。   |
| type          | text   | 知識類別：區分這筆是告警描述、策略建議、xApp說明等。 |
| tag           | text[] | 標籤，支援多個分類                                   |
| domain        | text   | 所屬領域：FlexRIC、CU、DU、Slice 等                  |
| source_doc    | text   | 來源文件，例如來自 PDF、Log、Spec 檔案的名稱         |
| created_at    | date   | 知識建立或插入的時間，支援查詢過濾                   |
| related_event | text   | 關聯的事件 ID（例如 ALARM_SINR_DROP）                |

**向量搜尋 :** 


| 用途               | 欄位                             |
|:------------------ | -------------------------------- |
| 語意搜尋（vector） | text（Ollama embedding 來源）    |
| 關鍵字過濾         | type, domain, tag, related_event |
| 時間查詢篩選       | created_at                       |

## 3. Backend API
### 3-1. rerank installation
- Utilize Hugging Face's rerank module - [Link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- Use the `bge-reranker-v2-m3`rerank module

**Step 0 :** Pre-request
python >=3.8
**Step 1 :** Install required libraries
```=
pip install -U transformers torch accelerate
```

### 3-2.  Installing and Using the Backend API
**Step 0 :** Pre-request
Run Weaviate with Docker
**Step 1 :** Install client library
```=
pip install fastapi uvicorn requests
pip install -r requirements.txt
```
**Step 1 :** Run
```=
uvicorn main:app --host 0.0.0.0 --port 8500
```

## 4. config
`main.py` : 
```=
OLLAMA_URL = <ollama server IP>
```









