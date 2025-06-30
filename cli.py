#!/usr/bin/env python3
"""
Command Line Interface for RAG Chatbot Testing
Provides comprehensive testing capabilities before API implementation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown

console = Console()

import dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.embeddings import GeminiEmbeddings
from src.core.llm import GeminiLLM
from src.core.vector_store import QdrantVectorStore
from src.core.retriever import SemanticRetriever
from src.data.indexer import DocumentIndexer
from src.rag.pipeline import RAGPipeline, RAGRequest, PipelineMode


class CLITester:
    """Command line interface for testing RAG components"""
    
    def __init__(self):
        """Initialize CLI tester"""
        # Load environment variables
        dotenv.load_dotenv()
        
        # Configure logging
        self._setup_logging()
        
        # Initialize components
        self.embeddings_service = None
        self.llm = None
        self.vector_store = None
        self.indexer = None
        self.pipeline = None
        
        # Configuration
        self.collection_name = os.getenv("COLLECTION_NAME", "test_documents")
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info("CLI Tester initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('cli_test.log')
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    async def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            print("🔧 Initializing components...")
            # Initialize embeddings service
            self.embeddings_service = GeminiEmbeddings(
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            await self.embeddings_service.health_check()
            print("✅ Embeddings service initialized")
            
            # Initialize LLM
            self.llm = GeminiLLM(
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            await self.llm.health_check()
            print("✅ LLM initialized")
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            print("✅ Vector store initialized")
            
            # Initialize indexer
            self.indexer = DocumentIndexer(
                embeddings_service=self.embeddings_service,
                vector_store=self.vector_store,
                collection_name=self.collection_name
            )
            print("✅ Document indexer initialized")
            
            # Initialize RAG pipeline
            self.pipeline = RAGPipeline(
                embeddings_service=self.embeddings_service,
                llm=self.llm,
                vector_store=self.vector_store,
                collection_name=self.collection_name
            )
            print("✅ RAG pipeline initialized")
            
            print("🎉 All components initialized successfully!\n")
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            logger.error(f"Component initialization failed: {e}")
            sys.exit(1)
    
    async def health_check(self):
        """Perform comprehensive health check"""
        print("🏥 Performing health check...\n")
        
        results = {}
        
        # Check vector store
        print("Checking vector store connection...")
        try:
            is_healthy = await self.vector_store.health_check()
            results["vector_store"] = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            print(f"  Vector Store: {results['vector_store']}")
        except Exception as e:
            results["vector_store"] = f"❌ Error: {e}"
            print(f"  Vector Store: {results['vector_store']}")
        
        # Check embeddings service
        print("Checking embeddings service...")
        try:
            test_embedding = await self.embeddings_service.generate_embedding("Test query")
            results["embeddings"] = f"✅ Healthy (dimension: {len(test_embedding)})"
            print(f"  Embeddings: {results['embeddings']}")
        except Exception as e:
            results["embeddings"] = f"❌ Error: {e}"
            print(f"  Embeddings: {results['embeddings']}")
        
        # Check LLM
        print("Checking LLM service...")
        try:
            test_response = await self.llm.generate("Say 'Hello'")
            results["llm"] = "✅ Healthy"
            print(f"  LLM: {results['llm']}")
            print(f"    Test response: {test_response[:50]}...")
        except Exception as e:
            results["llm"] = f"❌ Error: {e}"
            print(f"  LLM: {results['llm']}")
        
        # Check RAG pipeline
        print("Checking RAG pipeline...")
        try:
            health_status = await self.pipeline.health_check()
            results["pipeline"] = "✅ Healthy"
            print(f"  Pipeline: {results['pipeline']}")
            print(f"    Components: {list(health_status.keys())}")
        except Exception as e:
            results["pipeline"] = f"❌ Error: {e}"
            print(f"  Pipeline: {results['pipeline']}")
        
        print("\n📊 Health Check Summary:")
        for component, status in results.items():
            print(f"  {component}: {status}")
        
        return results
    
    async def index_document(self, file_path: str):
        """Index a single document"""
        print(f"📚 Indexing document: {file_path}")
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return
        
        try:
            start_time = time.time()
            indexed_doc = await self.indexer.index_document(file_path)
            processing_time = time.time() - start_time
            
            print(f"✅ Document indexed successfully!")
            print(f"  Document ID: {indexed_doc.document_id}")
            print(f"  Chunks: {indexed_doc.chunk_count}")
            print(f"  Tokens: {indexed_doc.total_tokens}")
            print(f"  Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to index document: {e}")
            logger.error(f"Document indexing failed: {e}")
    
    async def index_directory(self, directory_path: str):
        """Index all documents in a directory"""
        print(f"📁 Indexing directory: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            print(f"❌ Directory not found: {directory_path}")
            return
        
        # Find all supported files
        supported_extensions = ['.txt', '.md', '.pdf', '.docx']
        files = []
        for ext in supported_extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        if not files:
            print("❌ No supported files found in directory")
            return
        
        print(f"Found {len(files)} files to index:")
        for file in files:
            print(f"  - {file.name}")
        
        try:
            start_time = time.time()
            indexed_docs = await self.indexer.index_documents([str(f) for f in files])
            processing_time = time.time() - start_time
            
            print(f"\n✅ Directory indexed successfully!")
            print(f"  Documents processed: {len(indexed_docs)}")
            print(f"  Total processing time: {processing_time:.2f}s")
            
            # Show stats
            stats = self.indexer.get_stats()
            print(f"  Total chunks: {stats['chunks_indexed']}")
            print(f"  Failed documents: {stats['failed_documents']}")
            
        except Exception as e:
            print(f"❌ Failed to index directory: {e}")
            logger.error(f"Directory indexing failed: {e}")
    
    async def query_simple(self, query: str, max_sources: int = 5):
        """Perform a simple query"""
        print(f"🔍 Simple Query: {query}")
        print("─" * 50)
        
        try:
            request = RAGRequest(
                query=query,
                mode=PipelineMode.SIMPLE,
                max_sources=max_sources
            )
            
            start_time = time.time()
            response = await self.pipeline.process_query(request)
            processing_time = time.time() - start_time
            
            console.print("📝 Answer:", style="bold green")
            console.print(Markdown(response.answer))
            print("\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                print(f"{i}. File: {source.get('file_path', 'Unknown')}")
                print(f"   Score: {source.get('score', 0):.3f}")
                print(f"   Preview: {source.get('content', '')[:100]}...")
                print()
            
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"📊 Sources used: {len(response.sources)}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            logger.error(f"Query processing failed: {e}")
    
    async def query_conversation(self, session_id: str = "test_session"):
        """Start an interactive conversation"""
        print(f"💬 Starting conversation session: {session_id}")
        print("Type 'quit' to exit, 'clear' to clear history\n")
        
        while True:
            try:
                query = input("🤔 You: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'clear':
                    self.pipeline.clear_session(session_id)
                    print("🧹 Conversation history cleared\n")
                    continue
                elif not query:
                    continue
                
                request = RAGRequest(
                    query=query,
                    session_id=session_id,
                    mode=PipelineMode.CONVERSATIONAL
                )
                
                start_time = time.time()
                response = await self.pipeline.process_query(request)
                processing_time = time.time() - start_time
                
                # Print bot response with markdown formatting
                console.print("🤖 Bot:", style="bold cyan")
                console.print(Markdown(response.answer))
                print(f"⏱️  ({processing_time:.2f}s, {len(response.sources)} sources)\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    async def test_retrieval(self, query: str, limit: int = 10):
        """Test document retrieval without LLM generation"""
        print(f"🔎 Testing retrieval for: {query}")
        print("─" * 50)
        
        try:
            # Test direct retrieval
            retriever = SemanticRetriever(
                embeddings_service=self.embeddings_service,
                vector_store=self.vector_store,
                collection_name=self.collection_name
            )
            
            start_time = time.time()
            results = retriever.retrieve(query)
            retrieval_time = time.time() - start_time
            
            print(f"📊 Retrieved {len(results)} documents in {retrieval_time:.2f}s")
            print("\n📚 Results:")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result.score:.3f}")
                print(f"   File: {result.metadata.get('file_path', 'Unknown')}")
                print(f"   Content: {result.content[:150]}...")
                print()
        
        except Exception as e:
            print(f"❌ Retrieval test failed: {e}")
            logger.error(f"Retrieval test failed: {e}")
    
    async def show_stats(self):
        """Show system statistics"""
        print("📊 System Statistics")
        print("═" * 50)
        
        try:
            # Vector store stats
            print("🗃️  Vector Store:")
            if self.vector_store.collection_exists(self.collection_name):
                collection_info = self.vector_store.get_collection_info(self.collection_name)
                stats = self.vector_store.get_collection_stats(self.collection_name)
                if collection_info and stats:
                    print(f"  Collection: {self.collection_name}")
                    print(f"  Vector count: {stats.get('vectors_count', 0):,}")
                    print(f"  Indexed size: {stats.get('indexed_vectors_count', 0):,}")
            else:
                print(f"  Collection '{self.collection_name}' does not exist")
            
            # Indexer stats
            print("\n📚 Indexer:")
            indexer_stats = self.indexer.get_stats()
            for key, value in indexer_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:,}")
            
            # Pipeline stats
            print("\n🔄 Pipeline:")
            pipeline_stats = self.pipeline.get_stats()
            for key, value in pipeline_stats.items():
                if key == "mode_usage":
                    print(f"  {key}:")
                    for mode, count in value.items():
                        print(f"    {mode.value}: {count}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:,}")
                    
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
    
    async def list_documents(self):
        """List all indexed documents"""
        print("📋 Indexed Documents")
        print("═" * 50)
        
        try:
            documents = self.indexer.list_indexed_documents()
            
            if not documents:
                print("No documents indexed yet.")
                return
            
            print(f"Found {len(documents)} indexed documents:\n")
            
            for doc in documents:
                print(f"📄 {doc.file_path}")
                print(f"   ID: {doc.document_id}")
                print(f"   Chunks: {doc.chunk_count}")
                print(f"   Tokens: {doc.total_tokens:,}")
                print(f"   Indexed: {doc.indexed_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Processing time: {doc.processing_time_seconds:.2f}s")
                print()
                
        except Exception as e:
            print(f"❌ Failed to list documents: {e}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI Tester")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    subparsers.add_parser("health", help="Perform health check")
    
    # Index commands
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_group = index_parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("--file", "-f", help="Index a single file")
    index_group.add_argument("--directory", "-d", help="Index all files in directory")
    
    # Query commands
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--max-sources", "-m", type=int, default=5, help="Maximum sources to retrieve")
    
    # Conversation command
    conv_parser = subparsers.add_parser("chat", help="Start interactive conversation")
    conv_parser.add_argument("--session", "-s", default="test_session", help="Session ID")
    
    # Retrieval test command
    retrieval_parser = subparsers.add_parser("retrieve", help="Test document retrieval")
    retrieval_parser.add_argument("text", help="Query text")
    retrieval_parser.add_argument("--limit", "-l", type=int, default=10, help="Number of documents to retrieve")
    
    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")
    
    # List documents command
    subparsers.add_parser("list", help="List indexed documents")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI tester
    cli = CLITester()
    await cli._initialize_components()
    
    # Execute command
    try:
        if args.command == "health":
            await cli.health_check()
        
        elif args.command == "index":
            if args.file:
                await cli.index_document(args.file)
            elif args.directory:
                await cli.index_directory(args.directory)
        
        elif args.command == "query":
            await cli.query_simple(args.text, args.max_sources)
        
        elif args.command == "chat":
            await cli.query_conversation(args.session)
        
        elif args.command == "retrieve":
            await cli.test_retrieval(args.text, args.limit)
        
        elif args.command == "stats":
            await cli.show_stats()
        
        elif args.command == "list":
            await cli.list_documents()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Command failed: {e}")
        logger.error(f"Command execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 