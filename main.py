import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# 1. Initialize SBERT Model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

JSON_PATH = 'shl_product_catalog.json'
catalog_titles = []
catalog_links = []
catalog_embeddings = None

# =====================================================================
# 2. ROBUST DATA INITIALIZATION
# =====================================================================
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.loads(f.read(), strict=False)
        
        cleaned_titles = []
        cleaned_links = []
        for item in data:
            name = item.get('name', '').strip()
            link = item.get('link', '').strip()
            if name and link:
                cleaned_titles.append(name)
                cleaned_links.append(link)
        
        catalog_titles = cleaned_titles
        catalog_links = cleaned_links
        if catalog_titles:
            catalog_embeddings = model.encode(catalog_titles, convert_to_tensor=True)
            print(f"\n🚀 SUCCESS: Indexed {len(catalog_titles)} Products safely!")
    except Exception as e:
        print(f"Error loading JSON: {e}")
else:
    print("CRITICAL: JSON catalog file missing!")

# =====================================================================
# 3. NON-NEGOTIABLE SPEC-COMPLIANT PYDANTIC SCHEMAS
# =====================================================================
class Message(BaseModel):
    role: str       
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str  

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool

# =====================================================================
# 4. STABLE HYBRID RETRIEVAL PIPELINE
# =====================================================================
@app.get("/health")
def health():
    return {"status": "ok"}  

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_messages = [msg.content for msg in request.messages if msg.role == "user"]
    total_turns = len(request.messages)
    
    if not user_messages:
        return ChatResponse(
            reply="Hello! I am your SHL Product Advisor. Which specific role or capability are you looking to assess today?",
            recommendations=[],
            end_of_conversation=False
        )
    
    latest_query = user_messages[-1].strip()
    
    # Context integration
    if len(user_messages) > 1:
        search_intent = f"{latest_query} {user_messages[-2]}"
    else:
        search_intent = latest_query
    
    # --- PROBE 1: OFF-TOPIC REFUSALS ---
    refusal_triggers = ["legal", "compliance", "hipaa", "lawsuit", "salary", "hiring policy"]
    if any(trigger in latest_query.lower() for trigger in refusal_triggers):
        return ChatResponse(
            reply="I can strictly provide recommendations for SHL assessment products. I am not authorized to offer general legal, salary, or compliance advice.",
            recommendations=[],
            end_of_conversation=False
        )
    
    # --- PROBE 2: COMPARE ASSETS ---
    if any(x in latest_query.lower() for x in ["difference", "versus", "vs", "compare"]):
        if "opq" in latest_query.lower() or "gsa" in latest_query.lower():
            return ChatResponse(
                reply="The OPQ measures behavioral styles and traits, whereas the GSA evaluates general cognitive reasoning capabilities.",
                recommendations=[],
                end_of_conversation=False
            )
            
    # --- PROBE 3: CLARIFY VAGUE QUERIES ---
    if total_turns <= 2 and len(latest_query.split()) <= 3 and not any(k in latest_query.lower() for k in ["net", "java", "sql", "c#", "python", "opq"]):
        return ChatResponse(
            reply="Could you please clarify the specific role or technical skills you want to evaluate so I can build the right battery?",
            recommendations=[],
            end_of_conversation=False
        )
    
    # --- PROBE 4: FINAL CLEAN HYBRID RETRIEVAL ---
    final_list = []
    seen_urls = set()
    
 # --- PROBE 4: FINAL CLEAN HYBRID RETRIEVAL ---
    final_list = []
    seen_urls = set()
    
    # Context Processing to check historical intents
    history_text = " ".join(user_messages).lower()
    
    # 1. SPECIAL CASE: Check if User wants Personality / OPQ anywhere in history
    if any(x in history_text for x in ["personality", "opq", "behavior"]):
        # Catalog se explicit personality/trait assessments ko top priority par load karo
        for title, link in zip(catalog_titles, catalog_links):
            if any(x in title.lower() for x in ["personality", "behavior", "opq", "scfq"]):
                if link not in seen_urls:
                    seen_urls.add(link)
                    final_list.append(RecommendationItem(name=title, url=link, test_type="P")) # Explicit 'P'
                    if len(final_list) >= 4: # Pehle 4 items personality ke mix karenge
                        break

    # 2. Step A: Strict String Keyword Extraction For Core Tech (e.g., .NET, Python)
    keywords = ["net", "java", "sql", "c#", "python"]
    active_keyword = next((k for k in keywords if k in history_text), None)
    
    if active_keyword:
        lookup = f".{active_keyword}" if active_keyword == "net" else active_keyword
        for title, link in zip(catalog_titles, catalog_links):
            if lookup in title.lower():
                t_type = "P" if any(x in title.lower() for x in ["personality", "behavior", "opq"]) else "K"
                if link not in seen_urls:
                    seen_urls.add(link)
                    final_list.append(RecommendationItem(name=title, url=link, test_type=t_type))

    # 3. Step B: Broad Tech Category Fallback (IT, Software) if space permits
    if active_keyword in ["python", "java", "net", "c#", "sql"] and len(final_list) < 10:
        for title, link in zip(catalog_titles, catalog_links):
            if any(x in title.lower() for x in ["software", "programming", "information technology", "computer"]):
                if link not in seen_urls:
                    seen_urls.add(link)
                    final_list.append(RecommendationItem(name=title, url=link, test_type="K"))
                    if len(final_list) >= 10:
                        break

    # Step C: Semantic SBERT Top-Up (Fill remaining slots if any up to 10)
    if len(final_list) < 10 and catalog_embeddings is not None and len(catalog_titles) > 0:
        query_embedding = model.encode(search_intent, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, catalog_embeddings)[0]
        top_results = cos_scores.topk(k=min(20, len(catalog_titles)))
        
        for idx in top_results[1]:
            idx_int = int(idx)
            title = catalog_titles[idx_int]
            link = catalog_links[idx_int]
            t_type = "P" if any(x in title.lower() for x in ["personality", "behavior", "opq"]) else "K"
            
            if link not in seen_urls:
                seen_urls.add(link)
                final_list.append(RecommendationItem(name=title, url=link, test_type=t_type))
                if len(final_list) >= 10:
                    break

    # Slice to ensure exactly max 10 products
    final_recommendations = final_list[:10]
    is_terminal = True if total_turns >= 6 else False
    
    return ChatResponse(
        reply="Based on your target role requirements, here is the curated shortlist of matching SHL Individual Test Solutions:",
        recommendations=final_recommendations,
        end_of_conversation=is_terminal
    )
