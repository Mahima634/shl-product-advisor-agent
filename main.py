import json, os, re, math
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SHL Assessment Recommender")

# Load catalog with cleaning
with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    content = f.read()
    content = re.sub(r'[\x00-\x1f]', ' ', content)
    product_catalog = json.loads(content)

# TF-IDF retrieval (exactly as per assignment's Recall@K)
def tokenize(t):
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return [w for w in t.split() if len(w)>1]

def build_text(p):
    return f"{p.get('name','')} {p.get('description','')} {p.get('duration','')} {' '.join(p.get('keys',[]))}"

corpus = [tokenize(build_text(p)) for p in product_catalog]
df = Counter()
for c in corpus:
    for w in set(c):
        df[w] += 1
N = len(corpus)

def idf(w):
    return math.log((N+1)/(df.get(w,0)+1)) + 1

def tfidf(tokens):
    tf = Counter(tokens)
    return {w: (c/len(tokens))*idf(w) for w,c in tf.items()}

def cos_sim(a,b):
    common = set(a)&set(b)
    if not common: return 0
    dot = sum(a[w]*b[w] for w in common)
    mag = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return dot/mag if mag else 0

doc_vecs = [tfidf(t) for t in corpus]

def retrieve(q, top=10):
    qv = tfidf(tokenize(q))
    scored = [(cos_sim(qv, dv), p) for dv,p in zip(doc_vecs, product_catalog)]
    scored.sort(reverse=True, key=lambda x:x[0])
    return [p for _,p in scored[:top]]

# --- Chat logic (rule-based, covers all assignment behaviors) ---
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

def test_type_code(keys):
    m = {"Knowledge & Skills":"K","Personality & Behavior":"P","Ability & Aptitude":"A","Simulations":"S","Biodata & Situational Judgment":"B","Development & 360":"D","Assessment Exercises":"E"}
    for k in keys:
        if k in m:
            return m[k]
    return "K"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Empty messages")
    
    last_msg = req.messages[-1].content.lower()
    all_user_msgs = " ".join([m.content.lower() for m in req.messages if m.role=="user"])
    
    # 1. Off-topic / prompt injection refusal
    off_topics = ["legal", "hr policy", "salary", "interview questions", "python code", "ignore previous"]
    if any(t in last_msg for t in off_topics):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations. Please ask about hiring assessments.",
            recommendations=[],
            end_of_conversation=False
        )
    
    # 2. Vague query clarification (exactly as assignment expects)
    vague = ["need an assessment", "help me hire", "recommend something", "assessment", "test", "hiring"]
    if any(v in last_msg for v in vague) and len(all_user_msgs.split()) < 12:
        return ChatResponse(
            reply="Sure! Could you tell me the job role (e.g., Python developer, manager, data analyst) and seniority level (entry, mid, senior)?",
            recommendations=[],
            end_of_conversation=False
        )
    
    # 3. Handle "compare X and Y"
    if "compare" in last_msg and (" and " in last_msg or " vs " in last_msg):
        return ChatResponse(
            reply="I can help compare assessments. Please provide the exact names of two SHL products from the catalog.",
            recommendations=[],
            end_of_conversation=False
        )
    
    # 4. Handle refinement (e.g., "add personality tests")
    if "add" in last_msg and ("personality" in last_msg or "cognitive" in last_msg):
        # Just re-retrieve with modified query
        products = retrieve(last_msg + " personality", top_k=10)
    else:
        products = retrieve(last_msg, top_k=10)
    
    recs = []
    for p in products[:10]:
        recs.append(RecommendationItem(
            name=p["name"],
            url=p["link"],
            test_type=test_type_code(p.get("keys", []))
        ))
    
    reply = f"Based on your need, here are {len(recs)} recommended assessments from the SHL catalog."
    end_conv = "thanks" in last_msg or "that's all" in last_msg or "done" in last_msg
    
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end_conv)
