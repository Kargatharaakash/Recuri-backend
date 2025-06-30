import os
import json
import numpy as np
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    print("Warning: sentence-transformers not found. Embedding features disabled.")
    HAS_EMBEDDINGS = False

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
    "PINECONE_CLOUD", "PINECONE_REGION", "PINECONE_INDEX_NAME"
]:
    val = os.getenv(var)
    if val:
        if "KEY" in var:
            print(f"{var}={'*' * (len(val)-6) + val[-6:]}")
        else:
            print(f"{var}={val}")

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
        if HAS_GENAI:
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

        # Embeddings
        if HAS_EMBEDDINGS:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None

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
        else:
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

    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        if self.embedding_model:
            return self.embedding_model.encode(query)
        return None

    def find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)
        # Try Pinecone first
        if self.index and query_embedding is not None:
            try:
                print(f"[Pinecone] Querying for similar query: {query}")
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=1,
                    include_metadata=True
                )
                print(f"[Pinecone] Query response: {results}")
                if results.matches and results.matches[0].score >= threshold:
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
                return content
        else:
            return content

    def save_result(self, query: str, result: str):
        query_embedding = self.get_query_embedding(query)
        # Try Pinecone first
        if self.index and query_embedding is not None:
            try:
                print(f"[Pinecone] Upserting query: {query}")
                upsert_response = self.index.upsert([
                    {
                        'id': f"query_{hash(query)}",
                        'values': query_embedding.tolist(),
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
    return WebQueryAgent()# Pinecone
try:
    import pinecone
    HAS_PINECONE = True
except ImportError:
    print("Warning: pinecone-client not found. Pinecone features disabled.")
    HAS_PINECONE = False

# Gemini
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    print("Warning: google-generativeai package not found. Using fallback methods.")
    HAS_GENAI = False

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
    "PINECONE_CLOUD", "PINECONE_REGION", "PINECONE_INDEX_NAME"
]:
    val = os.getenv(var)
    if val:
        if "KEY" in var:
            print(f"{var}={'*' * (len(val)-6) + val[-6:]}")
        else:
            print(f"{var}={val}")

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
        if HAS_GENAI:
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

        # Embeddings
        if HAS_EMBEDDINGS:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None

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
        else:
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

    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        if self.embedding_model:
            return self.embedding_model.encode(query)
        return None

    def find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)
        # Try Pinecone first
        if self.index and query_embedding is not None:
            try:
                print(f"[Pinecone] Querying for similar query: {query}")
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=1,
                    include_metadata=True
                )
                print(f"[Pinecone] Query response: {results}")
                if results.matches and results.matches[0].score >= threshold:
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
                return content
        else:
            return content

    def save_result(self, query: str, result: str):
        query_embedding = self.get_query_embedding(query)
        # Try Pinecone first
        if self.index and query_embedding is not None:
            try:
                print(f"[Pinecone] Upserting query: {query}")
                upsert_response = self.index.upsert([
                    {
                        'id': f"query_{hash(query)}",
                        'values': query_embedding.tolist(),
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
