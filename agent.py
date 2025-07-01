import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    print("Warning: pinecone (v3) not found. Pinecone features disabled.")
    HAS_PINECONE = False

# Gemini
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    print("Warning: google-generativeai package not found. Using fallback methods.")
    HAS_GENAI = False

# Groq
try:
    import groq
    HAS_GROQ = True
except ImportError:
    print("Warning: groq package not found. Groq LLM support disabled.")
    HAS_GROQ = False

# Load .env.local first if present, then .env
if os.path.exists('.env.local'):
    print("Loaded environment variables from .env.local (if present):")
    load_dotenv('.env.local', override=True)
else:
    load_dotenv()
    print("Loaded environment variables from .env (if present):")

# Print all relevant env vars for debugging
for var in [
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "PINECONE_API_KEY",
    "PINECONE_CLOUD", "PINECONE_REGION", "PINECONE_INDEX_NAME", "HF_TOKEN", "GROQ_API_KEY"
]:
    val = os.getenv(var)
    if val:
        if "KEY" in var or "TOKEN" in var:
            print(f"{var}={'*' * (len(val)-6) + val[-6:]}")
        else:
            print(f"{var}={val}")

def get_query_embedding(query: str):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("⚠️ HF_TOKEN not set in environment. Please set it to use Hugging Face embedding API.")
        return None

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Use the feature-extraction pipeline endpoint, and send a single string
    API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": query}, timeout=30)
        if response.status_code == 200:
            emb = response.json()
            if isinstance(emb, list) and isinstance(emb[0], list):  # sometimes [[...]]
                emb = emb[0]
            if not isinstance(emb, list) or not all(isinstance(x, (float, int)) for x in emb):
                print("⚠️ Unexpected format of embedding")
                return None
            return emb
        else:
            try:
                err_json = response.json()
                if isinstance(err_json, dict) and err_json.get("error"):
                    print(f"⚠️ Hugging Face API error: {err_json.get('error')}")
                else:
                    print("⚠️ Hugging Face error response:", err_json)
            except:
                print("⚠️ Raw error:", response.text)
            return None
    except Exception as e:
        print(f"⚠️ Exception during Hugging Face embedding call: {e}")
        return None

class WebQueryAgent:
    def __init__(self):
        # Gemini
        print("Loaded environment variables from .env.local (if present):")
        dotenv_path = os.path.abspath('.env.local')
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        print(line)
        else:
            print("No .env.local file found.")
        # Gemini (DISABLED if GROQ is available)
        self.gemini_model = None
        if HAS_GROQ:
            print("ℹ️ Skipping Gemini: Groq is available and will be used for all LLM tasks.")
        elif HAS_GENAI:
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini API initialized")
            else:
                print("⚠️ No API key found for Gemini.")
                self.gemini_model = None
        else:
            self.gemini_model = None

        # Groq
        if HAS_GROQ:
            groq_api_key = os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_KEY') or os.environ.get('GROQ')
            if groq_api_key:
                self.groq_client = groq.Groq(api_key=groq_api_key)
                print("✅ Groq API initialized")
            else:
                print("⚠️ No API key found for Groq.")
                self.groq_client = None
        else:
            self.groq_client = None

        # Embeddings: Use Hugging Face API, so no local model needed
        # self.embedding_model = None

        # Pinecone (v3+)
        self.index = None
        if HAS_PINECONE:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            pinecone_cloud = os.getenv('PINECONE_CLOUD')
            pinecone_region = os.getenv('PINECONE_REGION')
            pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'query-agent')
            print(f"[Pinecone] Connecting with API key: {'set' if pinecone_api_key else 'not set'}")
            print(f"[Pinecone] Cloud: {pinecone_cloud}, Region: {pinecone_region}, Index: {pinecone_index_name}")

            try:
                self.pc = Pinecone(api_key=pinecone_api_key)
                # List indexes
                index_names = self.pc.list_indexes().names()
                print(f"[Pinecone] Available indexes: {index_names}")
                # Create index if not exists
                if pinecone_index_name not in index_names:
                    print(f"[Pinecone] Creating index: {pinecone_index_name}")
                    self.pc.create_index(
                        name=pinecone_index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=pinecone_cloud,
                            region=pinecone_region
                        )
                    )
                self.index = self.pc.Index(pinecone_index_name)
                print(f"[Pinecone] Connected to index: {pinecone_index_name}")
            except Exception as e:
                print(f"[Pinecone] Setup failed: {e}. Using local fallback.")
                self.index = None

        # Local cache fallback (DISABLED)
        self.local_cache = None
        self.cache_file = None

    # Local cache methods are now disabled
    def _load_local_cache(self):
        pass

    def _save_local_cache(self):
        pass

    def classify_query(self, query: str) -> bool:
        # Use Groq LLM for validation if available
        if self.groq_client:
            try:
                prompt = f"""
You are a query validator. Respond with VALID or INVALID for the following user query.

A VALID query is:
- A clear question or search request
- Something that can be answered by searching the web
- Coherent and makes sense

An INVALID query is:
- Multiple unrelated commands or requests
- Personal tasks like "walk my pet, add apples to grocery"
- Nonsensical or gibberish text
- Commands that aren't search-related

Query: "{query}"

Respond with only "VALID" or "INVALID".
"""
                chat_completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a query validator. Respond with VALID or INVALID for the following user query."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                result = chat_completion.choices[0].message.content.strip().upper()
                return result == "VALID"
            except Exception as e:
                print(f"Groq classification failed: {e}")
                return self._heuristic_classify(query)
        # Fallback to Gemini if available
        if self.gemini_model:
            try:
                prompt = f"""
You are a query validator. Analyze the following user query and determine if it's a valid search query.

A VALID query is:
- A clear question or search request
- Something that can be answered by searching the web
- Coherent and makes sense

An INVALID query is:
- Multiple unrelated commands or requests
- Personal tasks like "walk my pet, add apples to grocery"
- Nonsensical or gibberish text
- Commands that aren't search-related

Query: "{query}"

Respond with only "VALID" or "INVALID".
"""
                response = self.gemini_model.generate_content(prompt)
                result = response.text.strip().upper()
                return result == "VALID"
            except Exception as e:
                print(f"Gemini classification failed: {e}")
                return self._heuristic_classify(query)
        # Fallback to heuristic
        return self._heuristic_classify(query)

    def _heuristic_classify(self, query: str) -> bool:
        if not query or len(query.strip()) < 3:
            return False
        task_indicators = ['add to', 'remind me', 'call', 'text', 'email', 'schedule', 'buy', 'walk my']
        if any(indicator in query.lower() for indicator in task_indicators):
            return False
        if ',' in query and any(word in query.lower() for word in ['and', 'then', 'also']):
            parts = query.split(',')
            if len(parts) > 1 and not any(p.lower().startswith(('what', 'who', 'how', 'when', 'where', 'why')) for p in parts):
                return False
        return True

    def get_query_embedding(self, query: str):
        # Always use Hugging Face API for embeddings
        return get_query_embedding(query)

    def find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)
        # If embedding is None, do not proceed to Pinecone
        if query_embedding is None:
            print("⚠️ No embedding generated for query; skipping Pinecone similarity search.")
            return None
        # Try Pinecone first
        if self.index:
            try:
                print(f"[Pinecone] Querying for similar query: {query}")
                results = self.index.query(
                    vector=query_embedding,
                    top_k=1,
                    include_metadata=True
                )
                print(f"[Pinecone] Query response: {results}")
                if hasattr(results, "matches") and results.matches and hasattr(results.matches[0], "score") and results.matches[0].score >= threshold:
                    print(f"[Pinecone] Found similar query with score {results.matches[0].score}")
                    return {
                        'query': results.matches[0].metadata.get('query'),
                        'result': results.matches[0].metadata.get('result'),
                        'similarity': results.matches[0].score
                    }
            except Exception as e:
                print(f"Pinecone search failed: {e}")

        # No local cache fallback: Pinecone only
        return None

    def scrape_web(self, query: str) -> str:
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            print("Searching web...")
            response = requests.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return f"Search failed with status code: {response.status_code}"
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            for result in soup.select('.result'):
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    results.append(f"• {title}\n{snippet}\n")
            if not results:
                return "No results found for your query."
            return "\n".join(results[:5])
        except Exception as e:
            print(f"Error during web search: {e}")
            return f"Error searching the web: {str(e)}"

    def summarize_content(self, query: str, content: str) -> str:
        # Try Gemini first
        if self.gemini_model:
            try:
                prompt = f"""
Based on the following web content, provide a comprehensive and helpful answer to the user's query.

User Query: "{query}"

Web Content:
{content}

Instructions:
- Provide a clear, well-structured answer
- Include the most relevant and useful information
- Keep it concise but comprehensive

Answer:
"""
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini summarization failed: {e}")
        # Fallback to Groq LLM if available
        if self.groq_client:
            try:
                prompt = f"""
Based on the following web content, provide a comprehensive and helpful answer to the user's query.

User Query: "{query}"

Web Content:
{content}

Instructions:
- Provide a clear, well-structured answer
- Include the most relevant and useful information
- Keep it concise but comprehensive

Answer:
"""
                chat_completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a helpful web content summarizer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"Groq summarization failed: {e}")
        # Fallback: return raw content
        return content

    def save_result(self, query: str, result: str):
        query_embedding = self.get_query_embedding(query)
        # If embedding is None, do not proceed to Pinecone
        if query_embedding is None:
            print("⚠️ No embedding generated for query; skipping Pinecone upsert.")
            return
        # Try Pinecone first
        if self.index:
            try:
                print(f"[Pinecone] Upserting query: {query}")
                upsert_response = self.index.upsert([
                    {
                        'id': f"query_{hash(query)}",
                        'values': query_embedding,
                        'metadata': {
                            'query': query,
                            'result': result
                        }
                    }
                ])
                print(f"[Pinecone] Upsert response: {upsert_response}")
                print("Result saved to Pinecone")
                return
            except Exception as e:
                print(f"Failed to save to Pinecone: {e}")

        # No local cache fallback: Pinecone only

    def process_query(self, query: str) -> str:
        print(f"Processing query: '{query}'")
        print("Classifying query...")
        if not self.classify_query(query):
            return "❌ This is not a valid search query. Please provide a clear question or search request."
        print("Query is valid")
        print("Checking for similar past queries...")
        similar_result = self.find_similar_query(query)
        if similar_result:
            print(f"Found similar query with {similar_result['similarity']:.2f} similarity")
            return f"📋 Found similar query: '{similar_result['query']}'\n\n{similar_result['result']}"
        print("No similar queries found, searching web...")
        content = self.scrape_web(query)
        if not content or len(content.strip()) < 50:
            return "❌ Failed to find relevant web content for your query."
        result = self.summarize_content(query, content)
        print("Saving result for future queries...")
        self.save_result(query, result)
        return f"🔍 Search Results for: '{query}'\n\n{result}"

def run_agent():
    return WebQueryAgent()
