# Recuri Web Query Agent — Full System Flow

This document explains the **end-to-end flow** of the Recuri Web Query Agent, covering both the frontend (Next.js/React) and backend (FastAPI + Playwright + LLMs + Pinecone).

---

## 🧑‍💻 1. User Interaction (Frontend)

- **User opens the chatbot UI** (Next.js, React, Tailwind).
- User types a query (e.g., "Top 10 AI tools for productivity") and hits send.
- The UI:
  - Shows an animated "typing" indicator (like ChatGPT).
  - Sends a POST request to `/api/query` on the backend (Render/Railway).

---

## 🌐 2. API Proxy (Frontend)

- The Next.js API route (`/api/query.ts`) proxies the request to the backend FastAPI endpoint.
- This allows for local development and CORS-free production.

---

## 🚀 3. Backend Query Flow (FastAPI + agent.py)

### a. **Validation**
- The backend receives the query.
- The agent uses Groq LLM (or Gemini fallback) to classify the query as VALID or INVALID.
- If INVALID, returns a message: "❌ This is not a valid search query..."

### b. **Similarity Search (Pinecone)**
- If VALID, the agent generates an embedding for the query (via Hugging Face Inference API).
- It checks Pinecone for a similar query (vector search).
- If a valid, non-error result is found, it is returned immediately.

### c. **Web Scraping (Playwright)**
- If no valid cached result, the agent launches Playwright (headless Chromium).
- It scrapes the first 5 pages of DuckDuckGo search results for the query.
- Extracts titles and links from the results.

### d. **Summarization (Groq LLM or Gemini)**
- The agent sends the scraped content + original query to Groq LLM (or Gemini fallback).
- The LLM generates a human-readable, markdown-formatted summary.

### e. **Result Storage**
- The agent saves the query, embedding, and result in Pinecone for future similarity search.

### f. **Response**
- The backend returns the summary (or error) as `{ "result": ... }` to the frontend.

---

## 💬 4. Frontend Display

- The frontend receives the result.
- Uses `react-markdown` and `remark-gfm` to render the markdown with full styling (bold, lists, links, etc.).
- Shows the response in a chat bubble, with a copy-to-clipboard button.
- The UI remains responsive and animated, just like ChatGPT.

---

## 🛡️ 5. Health & Keepalive

- `/api/health` endpoint is used for uptime checks and keepalive scripts to prevent Render/Railway from sleeping.

---

## 🛠️ 6. Deployment Notes

- **Dockerfile** installs Playwright and Chromium for browser scraping.
- **render.yaml** or Railway config ensures Playwright is installed at build time.
- All environment variables (API keys, tokens) are loaded and masked in logs.

---

## 📊 7. Error Handling

- If Playwright or web scraping fails, the agent falls back to requests+BeautifulSoup.
- If all scraping fails, a clear error is returned and not cached in Pinecone.
- Only valid, non-error results are cached and returned for similar queries.

---

## 🧩 8. Extensibility

- The agent is modular: you can swap out LLMs, search engines, or vector DBs as needed.
- The frontend is ready for further UI/UX enhancements (e.g., streaming, avatars, etc.).

---

**This flow ensures a robust, production-ready, and user-friendly web query agent, meeting all Ripplica task requirements.**
