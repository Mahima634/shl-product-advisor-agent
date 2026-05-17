import json, os, re, math
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load catalog
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "shl_product_catalog.json")

try:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(r'[\x00-\x1f]', ' ', content)
        product_catalog = json.loads(content)
    print(f"✅ Catalog loaded: {len(product_catalog)} products")
except Exception as e:
    product_catalog = []
    print(f"❌ Catalog error: {e}")

# TF-IDF retrieval
def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

def build_doc_text(p: dict) -> str:
    return " ".join([
        p.get("name", ""),
        p.get("description", ""),
        p.get("job_levels_raw", "").strip().strip(","),
        " ".join(p.get("keys", [])),
        p.get("duration", ""),
    ])

corpus_tokens = [tokenize(build_doc_text(p)) for p in product_catalog]
N = len(corpus_tokens)

df_counts = Counter()
for tokens in corpus_tokens:
    for token in set(tokens):
        df_counts[token] += 1

def idf(term: str) -> float:
    df = df_counts.get(term, 0)
    if df == 0:
        return 0.0
    return math.log((N + 1) / (df + 1)) + 1.0

def tfidf_vector(tokens: list) -> dict:
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: (count / total) * idf(term) for term, count in tf.items()}

def cosine_sim(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    mag = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return dot / mag if mag else 0.0

doc_vectors = [tfidf_vector(tokens) for tokens in corpus_tokens]

def retrieve_products(query: str, top_k: int = 10) -> list:
    qvec = tfidf_vector(tokenize(query))
    scored = [(cosine_sim(qvec, dvec), p) for dvec, p in zip(doc_vectors, product_catalog)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]

def get_test_type_code(keys: list) -> str:
    mapping = {
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Ability & Aptitude": "A",
        "Simulations": "S",
        "Biodata & Situational Judgment": "B",
        "Development & 360": "D",
        "Assessment Exercises": "E",
    }
    for k in keys:
        if k in mapping:
            return mapping[k]
    return "K"

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool

# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    last_msg = req.messages[-1].content.lower()
    all_user_msgs = " ".join([m.content.lower() for m in req.messages if m.role == "user"])
    
    # Off-topic refusal
    off_topics = ["legal", "hr policy", "salary", "interview questions", "python code", "ignore previous", "hr advice"]
    if any(t in last_msg for t in off_topics):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations. Please ask about hiring assessments for specific roles.",
            recommendations=[],
            end_of_conversation=False
        )
    
    # Vague query clarification
    vague = ["need an assessment", "help me hire", "recommend something", "assessment", "test", "hiring"]
    if any(v in last_msg for v in vague) and len(all_user_msgs.split()) < 12:
        return ChatResponse(
            reply="Sure! Could you tell me the job role (e.g., Python developer, manager, data analyst) and seniority level (entry, mid, senior)?",
            recommendations=[],
            end_of_conversation=False
        )
    
    # Compare handling
    if "compare" in last_msg and (" and " in last_msg or " vs " in last_msg):
        return ChatResponse(
            reply="Please provide the exact names of two SHL assessments from the catalog to compare.",
            recommendations=[],
            end_of_conversation=False
        )
    
    # Retrieve recommendations
    products = retrieve_products(last_msg, top_k=10)
    recs = []
    for p in products[:10]:
        recs.append(RecommendationItem(
            name=p["name"],
            url=p["link"],
            test_type=get_test_type_code(p.get("keys", []))
        ))
    
    reply = f"Based on your need, here are {len(recs)} recommended assessments from the SHL catalog:"
    end_conv = "thanks" in last_msg or "that's all" in last_msg or "done" in last_msg
    
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end_conv)
