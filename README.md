# CLI Testing Guide for RAG Chatbot

This guide provides step-by-step instructions for testing the RAG chatbot application via command line before implementing the API layer.

## Prerequisites

### 1. Environment Setup
Make sure you have the following set up:

- Python 3.8+ installed
- Dependencies installed: `pip install -r requirements.txt`
- Qdrant vector database running
- Google Gemini API key configured

### 2. Environment Variables
Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

Required variables:
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `QDRANT_HOST`: Qdrant server host (default: localhost)
- `QDRANT_PORT`: Qdrant server port (default: 6333)

### 3. Start Qdrant Database
If using Docker:
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Testing Workflow

### Step 1: Verify Setup
First, run the setup verification script:

```bash
python test_setup.py
```

This will check:
- ✅ Environment variables
- 📦 Python dependencies  
- 🗃️ Qdrant connection
- 📊 Setup summary

Expected output:
```
🚀 RAG Chatbot Setup Verification
==================================================
✅ Environment file loaded
🔧 Checking Environment Variables...
✅ GOOGLE_API_KEY: **********
✅ QDRANT_HOST: localhost
✅ QDRANT_PORT: 6333
...
🎉 All checks passed! Ready to test the application.
```

### Step 2: Component Health Check
Verify all RAG components are working:

```bash
python cli.py health
```

This performs comprehensive health checks:
- 🗃️ Vector store connection
- 🧠 Embeddings service
- 🤖 LLM service  
- 🔄 RAG pipeline

Expected output:
```
🔧 Initializing components...
✅ Embeddings service initialized
✅ LLM initialized
✅ Vector store initialized
✅ Document indexer initialized
✅ RAG pipeline initialized
🎉 All components initialized successfully!

🏥 Performing health check...
...
📊 Health Check Summary:
  vector_store: ✅ Healthy
  embeddings: ✅ Healthy (dimension: 768)
  llm: ✅ Healthy
  pipeline: ✅ Healthy
```

### Step 3: Index Test Documents
Index the sample documents for testing:

#### Index Single Document
```bash
python cli.py index --file test_docs/sample_product.md
```

Expected output:
```
📚 Indexing document: test_docs/sample_product.md
✅ Document indexed successfully!
  Document ID: doc_12345678_abcd1234
  Chunks: 8
  Tokens: 1,247
  Processing time: 3.45s
```

#### Index Entire Directory
```bash
python cli.py index --directory test_docs
```

Expected output:
```
📁 Indexing directory: test_docs
Found 3 files to index:
  - sample_product.md
  - shipping_policy.txt
  - faq.md

✅ Directory indexed successfully!
  Documents processed: 3
  Total processing time: 8.92s
  Total chunks: 24
  Failed documents: 0
```

### Step 4: Test Document Retrieval
Test the retrieval system without LLM generation:

```bash
python cli.py retrieve "iPhone camera features" --limit 5
```

Expected output:
```
🔎 Testing retrieval for: iPhone camera features
──────────────────────────────────────────────────
📊 Retrieved 5 documents in 0.34s

📚 Results:
1. Score: 0.892
   File: test_docs/sample_product.md
   Content: ### Camera System
- **Main Camera**: 48MP Main with 2x Telephoto option
- **Ultra Wide**: 13MP Ultra Wide camera...

2. Score: 0.756
   File: test_docs/sample_product.md
   Content: - **Video**: 4K video recording at 24 fps, 25 fps, 30 fps, or 60 fps...
```

### Step 5: Test Simple Queries
Perform single-turn queries:

```bash
python cli.py query "What are the camera specifications of iPhone 15 Pro Max?"
```

Expected output:
```
🔍 Simple Query: What are the camera specifications of iPhone 15 Pro Max?
──────────────────────────────────────────────────
📝 Answer:
The iPhone 15 Pro Max features an advanced camera system with:

**Main Cameras:**
- 48MP Main camera with 2x Telephoto option
- 13MP Ultra Wide camera  
- 12MP 5x Telephoto camera

**Front Camera:**
- 12MP TrueDepth camera

**Video Recording:**
- 4K video recording at 24 fps, 25 fps, 30 fps, or 60 fps

📚 Sources:
1. File: test_docs/sample_product.md
   Score: 0.892
   Preview: ### Camera System - **Main Camera**: 48MP Main with 2x Telephoto option - **Ultra Wide**: 13MP...

⏱️ Processing time: 2.34s
📊 Sources used: 2
```

### Step 6: Test Interactive Conversation
Start an interactive chat session:

```bash
python cli.py chat --session test_user_123
```

This starts an interactive conversation:
```
💬 Starting conversation session: test_user_123
Type 'quit' to exit, 'clear' to clear history

🤔 You: What's the price of iPhone 15 Pro Max?
🤖 Bot: The iPhone 15 Pro Max is available in three storage options:
- 256GB: $1,199
- 512GB: $1,399  
- 1TB: $1,599
⏱️ (1.87s, 1 sources)

🤔 You: What about shipping costs?
🤖 Bot: Based on our shipping policy:
- Standard shipping is FREE for orders over $50 (your iPhone qualifies!)
- Expedited shipping: $12.99
- Overnight shipping: $24.99
⏱️ (1.45s, 2 sources)

🤔 You: clear
🧹 Conversation history cleared

🤔 You: quit
👋 Goodbye!
```

### Step 7: Monitor System Statistics
Check system performance and usage:

```bash
python cli.py stats
```

Expected output:
```
📊 System Statistics
══════════════════════════════════════════════════
🗃️ Vector Store:
  Collection: test_documents
  Vector count: 24
  Indexed size: 24

📚 Indexer:
  documents_processed: 3
  chunks_indexed: 24
  total_processing_time: 8.92
  embedding_time: 6.45
  storage_time: 1.89
  duplicates_skipped: 0
  failed_documents: 0

🔄 Pipeline:
  total_requests: 5
  successful_requests: 5
  failed_requests: 0
  average_processing_time: 1.89
  active_sessions: 1
  mode_usage:
    simple: 3
    conversational: 2
    streaming: 0
```

### Step 8: List Indexed Documents
View all indexed documents:

```bash
python cli.py list
```

Expected output:
```
📋 Indexed Documents
══════════════════════════════════════════════════
Found 3 indexed documents:

📄 test_docs/sample_product.md
   ID: doc_12345678_abcd1234
   Chunks: 8
   Tokens: 1,247
   Indexed: 2024-01-15 14:30:25
   Processing time: 3.45s

📄 test_docs/shipping_policy.txt
   ID: doc_87654321_efgh5678
   Chunks: 7
   Tokens: 892
   Indexed: 2024-01-15 14:32:10
   Processing time: 2.78s

📄 test_docs/faq.md
   ID: doc_13579246_ijkl9012
   Chunks: 9
   Tokens: 1,456
   Indexed: 2024-01-15 14:32:15
   Processing time: 2.69s
```

## Advanced Testing Scenarios

### Test Error Handling
Try these scenarios to test error handling:

1. **Query with empty index:**
   ```bash
   # Delete collection and query
   python cli.py query "test query"
   ```

2. **Invalid file path:**
   ```bash
   python cli.py index --file nonexistent.txt
   ```

3. **Network connectivity issues:**
   ```bash
   # Stop Qdrant and run health check
   python cli.py health
   ```

### Performance Testing
Test with larger datasets:

1. **Large document indexing:**
   ```bash
   python cli.py index --directory /path/to/large/document/collection
   ```

2. **Batch queries:**
   ```bash
   # Run multiple queries in sequence
   python cli.py query "query 1"
   python cli.py query "query 2" 
   python cli.py query "query 3"
   ```

### Conversation Flow Testing
Test complex conversation patterns:

1. **Multi-turn context:**
   - Ask about a product
   - Ask follow-up questions
   - Reference previous answers
   - Clear history and verify reset

2. **Session management:**
   - Start multiple sessions with different IDs
   - Switch between sessions
   - Verify conversation isolation

## Troubleshooting

### Common Issues

1. **"Failed to initialize components"**
   - Check `.env` file configuration
   - Verify Qdrant is running
   - Check Google API key validity

2. **"Collection does not exist"**
   - Run indexing first: `python cli.py index --directory test_docs`
   - Check Qdrant connection

3. **"Embedding generation failed"**
   - Verify Google API key
   - Check internet connection
   - Check API quota/limits

4. **"Vector store connection failed"**
   - Ensure Qdrant is running on correct port
   - Check host/port configuration in `.env`

### Performance Issues

1. **Slow indexing:**
   - Reduce batch size in indexer configuration
   - Check network latency to Gemini API
   - Monitor memory usage

2. **Slow queries:**
   - Check vector collection size
   - Monitor Qdrant memory usage
   - Verify embedding generation time

### Debugging Tips

1. **Enable debug logging:**
   ```bash
   export LOG_LEVEL=DEBUG
   python cli.py health
   ```

2. **Check log files:**
   ```bash
   tail -f cli_test.log
   ```

3. **Monitor system resources:**
   - CPU usage during embedding generation
   - Memory usage during indexing
   - Disk space for vector storage

## Next Steps

After successful CLI testing:

1. ✅ All health checks pass
2. ✅ Documents indexed successfully  
3. ✅ Queries return relevant results
4. ✅ Conversations work properly
5. ✅ Performance is acceptable

You're ready to proceed with API implementation:
- Implement FastAPI routes (`src/api/routes.py`)
- Add authentication/authorization
- Create API documentation
- Deploy to production environment

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `health` | Health check all components | `python cli.py health` |
| `index --file` | Index single document | `python cli.py index --file doc.txt` |
| `index --directory` | Index all files in directory | `python cli.py index --directory ./docs` |
| `query` | Simple query | `python cli.py query "question"` |
| `chat` | Interactive conversation | `python cli.py chat --session user1` |
| `retrieve` | Test document retrieval | `python cli.py retrieve "search term"` |
| `stats` | Show system statistics | `python cli.py stats` |
| `list` | List indexed documents | `python cli.py list` | 