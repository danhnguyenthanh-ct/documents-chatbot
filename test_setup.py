#!/usr/bin/env python3
"""
Setup Verification Script
Checks environment variables, dependencies, and basic system requirements.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check environment variables"""
    print("🔧 Checking Environment Variables...")
    
    required_vars = [
        "GOOGLE_API_KEY",
        "QDRANT_HOST", 
        "QDRANT_PORT"
    ]
    
    optional_vars = [
        "COLLECTION_NAME",
        "LOG_LEVEL",
        "EMBEDDING_MODEL",
        "LLM_MODEL"
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(len(value), 10)}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"ℹ️  {var}: Using default")
    
    return missing_vars

def check_dependencies():
    """Check Python dependencies"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        "langchain",
        "langchain_google_genai", 
        "qdrant_client",
        "dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    return missing_packages

def check_qdrant_connection():
    """Check Qdrant connection"""
    print("\n🗃️  Checking Qdrant Connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=host, port=port, timeout=5.0)
        collections = client.get_collections()
        
        print(f"✅ Qdrant connection successful")
        print(f"   Host: {host}:{port}")
        print(f"   Collections: {len(collections.collections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print(f"   Make sure Qdrant is running on {host}:{port}")
        return False

def main():
    """Main setup check"""
    print("🚀 RAG Chatbot Setup Verification")
    print("=" * 50)
    
    # Load environment
    try:
        import dotenv
        dotenv.load_dotenv()
        print("✅ Environment file loaded")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
    
    # Check environment variables
    missing_vars = check_environment()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Check Qdrant connection
    qdrant_ok = check_qdrant_connection()
    
    # Summary
    print("\n📊 Setup Summary")
    print("-" * 30)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Create a .env file from env.example and set these variables")
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
    
    if not qdrant_ok:
        print("❌ Qdrant not accessible")
        print("   Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    
    if not missing_vars and not missing_packages and qdrant_ok:
        print("🎉 All checks passed! Ready to test the application.")
        print("\nNext steps:")
        print("1. Run: python cli.py health")
        print("2. Index documents: python cli.py index --directory ./test_docs")
        print("3. Test queries: python cli.py query 'your question here'")
    else:
        print("⚠️  Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main() 