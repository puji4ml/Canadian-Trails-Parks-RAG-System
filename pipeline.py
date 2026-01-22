#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Literal
import time
import json
import requests
from dataclasses import dataclass


# In[2]:


import os
os.environ["GROQ_API_KEY"] = "insert your key"


# In[3]:


# =============================================================================
# CONFIGURATION WITH FREE LLM OPTIONS
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for different RAG pipeline variants"""
    name: str
    collection_name: str
    llm_provider: Literal["ollama", "groq", "local", "openai", "anthropic"]
    llm_model: str
    top_k: int
    use_reranking: bool
    use_metadata_filtering: bool
    temperature: float
    max_tokens: int
    
    @property
    def is_free(self) -> bool:
        """Check if this pipeline uses free LLM"""
        return self.llm_provider in ["ollama", "groq", "local"]



# In[4]:


# =============================================================================
# FREE PIPELINE CONFIGURATIONS
# =============================================================================

FREE_PIPELINES = {
    "ollama_speed": PipelineConfig(
        name="🆓 Ollama Speed (Llama 3.2 - Local)",
        collection_name="small_minilm",
        llm_provider="ollama",
        llm_model="llama3.2:3b",  # 3B model, very fast
        top_k=3,
        use_reranking=False,
        use_metadata_filtering=True,
        temperature=0.3,
        max_tokens=500
    ),
    
    "ollama_quality": PipelineConfig(
        name="🆓 Ollama Quality (Llama 3.1 - Local)",
        collection_name="medium_minilm",
        llm_provider="ollama",
        llm_model="llama3.1:8b",  # 8B model, better quality
        top_k=5,
        use_reranking=True,
        use_metadata_filtering=True,
        temperature=0.2,
        max_tokens=1000
    ),
    
    "groq_fast": PipelineConfig(
        name="🆓 Groq Ultra-Fast (Llama 3.1 - Cloud)",
        collection_name="small_minilm",
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",  # Very fast API
        top_k=3,
        use_reranking=False,
        use_metadata_filtering=True,
        temperature=0.3,
        max_tokens=500
    ),
    
    
    "local_model": PipelineConfig(
        name="🆓 Local Model (Tiny Llama - No Internet)",
        collection_name="small_minilm",
        llm_provider="local",
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        top_k=3,
        use_reranking=False,
        use_metadata_filtering=True,
        temperature=0.5,
        max_tokens=500
    ),
    
    "ollama_quality_xlarge": PipelineConfig(
        name="🆓 Ollama Quality - XLarge (Llama 3.1)",
        collection_name="extra_large_minilm",
        llm_provider="ollama",
        llm_model="llama3.1:8b",
        top_k=5,
        use_reranking=True,
        use_metadata_filtering=True,
        temperature=0.2,
        max_tokens=1000
    ),
    
    "groq_fast_xlarge": PipelineConfig(
        name="🆓 Groq Fast - XLarge (Llama 3.1)",
        collection_name="extra_large_minilm",
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        top_k=5,
        use_reranking=False,
        use_metadata_filtering=True,
        temperature=0.3,
        max_tokens=800
    )
}




# In[5]:


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridRetriever:
    """Advanced retrieval with metadata filtering"""
    
    def __init__(self, collection_name: str, db_path: str = "data/vector_db"):
        db_path = f"{db_path}/{collection_name}"
        print(f"📁 loading collection from: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.location_mapping = {
            "toronto": ["Ontario South", "Ontario Central"],
            "vancouver": ["British Columbia South"],
            "montreal": ["Quebec South"],
            "calgary": ["Alberta South"],
            "ottawa": ["Ontario South"],
            "banff": ["Alberta South", "AB"],
        }
    
    def extract_location_filter(self, query: str) -> Optional[List[str]]:
        """Extract location from query"""
        query_lower = query.lower()
        
        for city, regions in self.location_mapping.items():
            if city in query_lower:
                print(f"🗺️  Detected location: {city} -> {regions}")
                return regions
        
        provinces = {
            "ontario": ["Ontario South", "Ontario Central", "Ontario North"],
            "quebec": ["Quebec South", "Quebec North"],
            "british columbia": ["British Columbia South", "British Columbia North"],
            "alberta": ["Alberta South", "Alberta North"],
            "bc": ["British Columbia South", "British Columbia North"],
        }
        
        for province, regions in provinces.items():
            if province in query_lower:
                print(f"🗺️  Detected province: {province} -> {regions}")
                return regions
        
        return None
    
    def retrieve(self, query: str, top_k: int = 5, use_metadata_filtering: bool = True) -> Dict:
        """Hybrid retrieval with metadata filtering"""
        start_time = time.time()
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        where_filter = None
        if use_metadata_filtering:
            regions = self.extract_location_filter(query)
            if regions:
                where_filter = {"region": {"$in": regions}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            where=where_filter
        )
        
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'rank': i + 1,
                'content': doc,
                'metadata': metadata,
                'similarity': 1 - distance,
                'distance': distance
            })
        
        retrieval_time = time.time() - start_time
        
        return {
            'results': formatted_results[:top_k],
            'retrieval_time': retrieval_time,
            'query': query,
            'filters_applied': where_filter
        }




# In[6]:


# =============================================================================
# FREE LLM GENERATORS
# =============================================================================

class FreeLLMGenerator:
    """
    LLM Generator supporting FREE models:
    - Ollama (run locally)
    - HuggingFace Inference API (free tier)
    - Groq (free tier, very fast)
    - Local models via transformers
    """
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        
        if provider == "ollama":
            self.base_url = "http://localhost:11434"
            print(f"🦙 Using Ollama: {model}")
            
        elif provider == "groq":
            self.api_key = self._get_groq_key()
            print(f"⚡ Using Groq: {model}")
            
            
        elif provider == "local":
            self._init_local_model()
            print(f"💻 Using Local Model: {model}")
    
    def _get_groq_key(self) -> str:
        """Get Groq API key (free tier available)"""
        import os
        key = os.getenv("GROQ_API_KEY")
        if not key:
            print("⚠️  Set GROQ_API_KEY environment variable")
            print("   Get free key at: https://console.groq.com")
        return key
    
    
    def _init_local_model(self):
        """Initialize local model using transformers"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("📥 Loading local model (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.local_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("✅ Local model loaded")
    
    def generate(self, query: str, context: List[Dict], temperature: float = 0.3, max_tokens: int = 800) -> Dict:
        """Generate answer using free LLM"""
        
        context_str = self._build_context(context)
        prompt = self._build_prompt(query, context_str)
        
        start_time = time.time()
        
        if self.provider == "ollama":
            response = self._generate_ollama(prompt, temperature, max_tokens)
        elif self.provider == "groq":
            response = self._generate_groq(prompt, temperature, max_tokens)
        elif self.provider == "local":
            response = self._generate_local(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        generation_time = time.time() - start_time
        
        return {
            'answer': response['text'],
            'generation_time': generation_time,
            'tokens_used': response.get('tokens', 0),
            'sources': [r['metadata'].get('name', 'Unknown') for r in context],
            'cost': 0.0  # FREE!
        }
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build formatted context"""
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            content = result['content'][:500]
            
            source_info = f"[Source {i}]"
            if 'name' in metadata:
                source_info += f" {metadata['name']}"
            if 'region' in metadata:
                source_info += f" - {metadata['region']}"
            
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt"""
        return f"""You are a helpful assistant for Canadian trails and parks information.

            Use the following context to answer the user's question. Cite sources using [Source N] format.

            Context:
            {context}

            User Question: {query}

            Instructions:
            - Provide a detailed answer based on the context
            - Always cite sources using [Source N] format
            - If context is insufficient, say so
            - Be specific about trail names and locations

            Answer:"""
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> Dict:
        """Generate using Ollama (local)"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                'text': result['response'],
                'tokens': result.get('eval_count', 0)
            }
        except requests.exceptions.ConnectionError:
            return {
                'text': "❌ Error: Ollama not running. Start it with: `ollama serve`",
                'tokens': 0
            }
        except Exception as e:
            return {
                'text': f"❌ Error: {str(e)}",
                'tokens': 0
            }
    
    def _generate_groq(self, prompt: str, temperature: float, max_tokens: int) -> Dict:
        """Generate using Groq (free, very fast)"""
        if not self.api_key:
            return {'text': "❌ Error: GROQ_API_KEY not set", 'tokens': 0}
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                'text': result['choices'][0]['message']['content'],
                'tokens': result['usage']['total_tokens']
            }
        except Exception as e:
            return {'text': f"❌ Error: {str(e)}", 'tokens': 0}
    
    
    def _generate_local(self, prompt: str, temperature: float, max_tokens: int) -> Dict:
        """Generate using local model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
        
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return {
            'text': response,
            'tokens': len(outputs[0])
        }




# In[7]:


# =============================================================================
# RAG PIPELINE WITH FREE LLMS
# =============================================================================

class FreeRAGPipeline:
    """RAG Pipeline using FREE LLMs"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.retriever = HybridRetriever(config.collection_name)
        self.generator = FreeLLMGenerator(config.llm_provider, config.llm_model)
        
        if config.use_reranking:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            self.reranker = None
    
    def query(self, question: str) -> Dict:
        """Complete RAG pipeline"""
        print(f"\n{'='*70}")
        print(f"🚀 {self.config.name}")
        print(f"{'='*70}")
        print(f"❓ Query: {question}\n")
        
        start_time = time.time()
        
        # Retrieve
        print("📥 Retrieving documents...")
        retrieval_results = self.retriever.retrieve(
            question,
            top_k=self.config.top_k,
            use_metadata_filtering=self.config.use_metadata_filtering
        )
        results = retrieval_results['results']
        print(f"   ✅ Retrieved {len(results)} results")
        
        # Rerank (optional)
        if self.reranker:
            print("🔄 Re-ranking...")
            pairs = [(question, r['content'][:500]) for r in results]
            scores = self.reranker.predict(pairs)
            for r, score in zip(results, scores):
                r['rerank_score'] = float(score)
            results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Generate
        print("💬 Generating answer...")
        generation_result = self.generator.generate(
            question,
            results,
            self.config.temperature,
            self.config.max_tokens
        )
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️  Total: {total_time:.2f}s | 💰 Cost: $0.00 (FREE!)")
        print(f"\n📊 Answer:\n{generation_result['answer']}\n")
        
        return {
            'query': question,
            'answer': generation_result['answer'],
            'sources': generation_result['sources'],
            'metrics': {
                'retrieval_time': retrieval_results['retrieval_time'],
                'generation_time': generation_result['generation_time'],
                'total_time': total_time,
                'tokens_used': generation_result['tokens_used'],
                'cost': 0.0
            },
            'config': {
                'pipeline': self.config.name,
                'llm': f"{self.config.llm_provider}/{self.config.llm_model}",
                'is_free': self.config.is_free
            }
        }




# In[8]:


# =============================================================================
# SETUP & TESTING
# =============================================================================

def check_free_llm_availability():
    """Check which free LLMs are available"""
    print("\n" + "="*70)
    print("🔍 Checking Free LLM Availability")
    print("="*70 + "\n")
    
    available = []
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama: Running with {len(models)} models")
            available.append("ollama")
        else:
            print("⚠️  Ollama: Not running (start with: `ollama serve`)")
    except:
        print("❌ Ollama: Not installed")
        print("   Install: https://ollama.com/download")
    
    # Check Groq
    import os
    if os.getenv("GROQ_API_KEY"):
        print("✅ Groq: API key found")
        available.append("groq")
    else:
        print("⚠️  Groq: No API key (get free key at console.groq.com)")
    
    
    # Check Local
    try:
        import torch
        print(f"✅ Local Models: PyTorch available (GPU: {torch.cuda.is_available()})")
        available.append("local")
    except:
        print("⚠️  Local Models: Install PyTorch and transformers")
    
    return available


def test_free_pipeline():
    """Quick test of free pipeline"""
    print("\n" + "="*70)
    print("🧪 Testing Free RAG Pipeline")
    print("="*70)
    
    available = check_free_llm_availability()
    
    if not available:
        print("\n❌ No free LLMs available!")
        print("\n📋 Setup Instructions:")
        print("   1. Install Ollama: https://ollama.com/download")
        print("   2. Run: `ollama pull llama3.2:3b`")
        print("   3. Run: `ollama serve`")
        return
    
    # Use the first available free LLM
    if "ollama" in available:
        config = FREE_PIPELINES["ollama_quality"]
    elif "groq" in available:
        config = FREE_PIPELINES["groq_fast"]

    
    pipeline = FreeRAGPipeline(config)
    result = pipeline.query("Find beginner-friendly trails near Toronto")
    
    print("\n✅ Test complete!")
    return result


if __name__ == "__main__":
    test_free_pipeline()


# ## Pipeline Testing - All Free models

# In[9]:


import json
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd


# In[10]:


TEST_QUERIES = [
    {
        "id": 1,
        "query": "What are hiking trails in British Columbia?",
        "category": "location",
        "difficulty": "easy",
        "expected": "Should find BC trails with location filtering"
    },
    {
        "id": 2,
        "query": "Find trails with concrete surface in Quebec",
        "category": "surface",
        "difficulty": "medium",
        "expected": "Should find Quebec trails with concrete surface"
    },
    {
        "id": 3,
        "query": "Are there wheelchair accessible trails in Ontario?",
        "category": "accessibility",
        "difficulty": "easy",
        "expected": "Should find wheelchair-accessible Ontario trails"
    },
    {
        "id": 4,
        "query": "What trails allow bicycles?",
        "category": "activity",
        "difficulty": "easy",
        "expected": "Should find trails that permit bicycles"
    },
    {
        "id": 5,
        "query": "Tell me about Banff National Park",
        "category": "park",
        "difficulty": "easy",
        "expected": "Should provide Banff park information"
    },
    {
        "id": 6,
        "query": "What are challenging mountain trails in Alberta?",
        "category": "difficulty",
        "difficulty": "hard",
        "expected": "Should find difficult/challenging Alberta trails"
    },
    {
        "id": 7,
        "query": "Find beginner-friendly trails near Toronto",
        "category": "multi-criteria",
        "difficulty": "medium",
        "expected": "Should find easy trails in Toronto area (THE PROBLEMATIC ONE)"
    }
]


# In[11]:


def test_single_model(pipeline_config, test_queries: List[Dict]) -> List[Dict]:
    """
    Test a single model on all queries
    """
    pipeline = FreeRAGPipeline(pipeline_config)
    results = []
    
    print(f"\n{'='*70}")
    print(f"Testing: {pipeline_config.name}")
    print(f"{'='*70}\n")
    
    for test in test_queries:
        print(f"[{test['id']}/7] {test['query'][:50]}...")
        
        try:
            start_time = time.time()
            result = pipeline.query(test['query'])
            elapsed = time.time() - start_time
            
            results.append({
                'query_id': test['id'],
                'query': test['query'],
                'category': test['category'],
                'answer': result['answer'],
                'sources': result['sources'],
                'retrieval_time': result['metrics']['retrieval_time'],
                'generation_time': result['metrics']['generation_time'],
                'total_time': result['metrics']['total_time'],
                'tokens': result['metrics']['tokens_used'],
                'answer_length': len(result['answer']),
                'num_sources': len(result['sources']),
                'success': True,
                'error': None
            })
            
            print(f"   ✅ Success ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'query_id': test['id'],
                'query': test['query'],
                'category': test['category'],
                'success': False,
                'error': str(e),
                'total_time': 0
            })
    
    return results


def test_all_models(test_queries: List[Dict] = TEST_QUERIES) -> Dict:
    """
    Test all available free models
    """
    print("\n" + "="*70)
    print("🧪 TESTING ALL FREE MODELS")
    print("="*70)
    
    # Check availability
    available_providers = check_free_llm_availability()
    
    if not available_providers:
        print("\n❌ No free models available!")
        print("\n📋 Setup instructions:")
        print("   1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("   2. Run: ollama serve && ollama pull llama3.2:3b")
        print("   3. Or get Groq API key: https://console.groq.com")
        return {}
    
    # Test each available pipeline
    all_results = {}
    
    for pipeline_name, config in FREE_PIPELINES.items():
        provider = config.llm_provider
        
        # Skip if provider not available
        if provider not in available_providers:
            print(f"\n⏭️  Skipping {config.name} ({provider} not available)")
            continue
        
        try:
            results = test_single_model(config, test_queries)
            all_results[pipeline_name] = {
                'config': {
                    'name': config.name,
                    'provider': config.llm_provider,
                    'model': config.llm_model,
                    'top_k': config.top_k,
                    'reranking': config.use_reranking
                },
                'results': results
            }
        except Exception as e:
            print(f"\n❌ Pipeline {pipeline_name} failed: {str(e)}")
            continue
    
    return all_results


# In[12]:


def analyze_results(all_results: Dict) -> Dict:
    """
    Analyze and compare results across all models
    """
    print("\n" + "="*70)
    print("📊 RESULTS ANALYSIS")
    print("="*70)
    
    if not all_results:
        print("\n❌ No results to analyze")
        return {}
    
    analysis = {
        'model_comparison': [],
        'query_comparison': [],
        'recommendations': {}
    }
    
    # Model-level analysis
    print("\n⏱️  SPEED COMPARISON")
    print("-" * 70)
    print(f"{'Model':<40} {'Avg Time':<12} {'Success Rate':<15} {'Avg Length'}")
    print("-" * 70)
    
    for pipeline_name, data in all_results.items():
        results = data['results']
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            continue
        
        avg_time = sum(r['total_time'] for r in successful) / len(successful)
        success_rate = len(successful) / len(results) * 100
        avg_length = sum(r['answer_length'] for r in successful) / len(successful)
        
        print(f"{data['config']['name']:<40} "
              f"{avg_time:.2f}s{'':<7} "
              f"{success_rate:.0f}%{'':<11} "
              f"{avg_length:.0f} chars")
        
        analysis['model_comparison'].append({
            'pipeline': pipeline_name,
            'name': data['config']['name'],
            'provider': data['config']['provider'],
            'avg_time': avg_time,
            'success_rate': success_rate,
            'avg_answer_length': avg_length,
            'total_queries': len(results),
            'successful_queries': len(successful)
        })
    
    # Query-level analysis
    print("\n📝 QUERY-LEVEL COMPARISON")
    print("-" * 70)
    
    for query_data in TEST_QUERIES:
        query_id = query_data['id']
        print(f"\n[Query {query_id}] {query_data['query']}")
        print(f"Category: {query_data['category']} | Difficulty: {query_data['difficulty']}")
        print("-" * 70)
        
        query_results = []
        
        for pipeline_name, data in all_results.items():
            result = next((r for r in data['results'] if r['query_id'] == query_id), None)
            
            if result and result.get('success'):
                print(f"  {data['config']['name'][:35]:<35} "
                      f"{result['total_time']:.2f}s | "
                      f"{result['answer_length']:>4} chars | "
                      f"{result['num_sources']} sources")
                
                query_results.append({
                    'pipeline': pipeline_name,
                    'time': result['total_time'],
                    'length': result['answer_length'],
                    'sources': result['num_sources']
                })
        
        if query_results:
            fastest = min(query_results, key=lambda x: x['time'])
            most_detailed = max(query_results, key=lambda x: x['length'])
            
            print(f"\n  ⚡ Fastest: {fastest['pipeline']} ({fastest['time']:.2f}s)")
            print(f"  📚 Most detailed: {most_detailed['pipeline']} ({most_detailed['length']} chars)")
    
    # Generate recommendations
    if analysis['model_comparison']:
        fastest_model = min(analysis['model_comparison'], key=lambda x: x['avg_time'])
        most_detailed = max(analysis['model_comparison'], key=lambda x: x['avg_answer_length'])
        
        analysis['recommendations'] = {
            'fastest': fastest_model,
            'most_detailed': most_detailed,
            'balanced': fastest_model if fastest_model['avg_time'] < 5 else most_detailed
        }
    
    return analysis


# In[13]:


def print_recommendations(analysis: Dict):
    """
    Print final recommendations
    """
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    if not analysis.get('recommendations'):
        print("\n❌ No recommendations available")
        return
    
    rec = analysis['recommendations']
    
    if 'fastest' in rec:
        print(f"\n⚡ FASTEST MODEL:")
        print(f"   {rec['fastest']['name']}")
        print(f"   Average time: {rec['fastest']['avg_time']:.2f}s")
        print(f"   Best for: Real-time queries, high-volume applications")
    
    if 'most_detailed' in rec:
        print(f"\n📚 MOST DETAILED MODEL:")
        print(f"   {rec['most_detailed']['name']}")
        print(f"   Average answer length: {rec['most_detailed']['avg_answer_length']:.0f} chars")
        print(f"   Best for: Comprehensive answers, research queries")
    
    if 'balanced' in rec:
        print(f"\n⚖️  RECOMMENDED FOR PRODUCTION:")
        print(f"   {rec['balanced']['name']}")
        print(f"   Time: {rec['balanced']['avg_time']:.2f}s")
        print(f"   Quality: {rec['balanced']['avg_answer_length']:.0f} chars avg")
    
    # Provider-specific recommendations
    ollama_models = [m for m in analysis['model_comparison'] if m['provider'] == 'ollama']
    groq_models = [m for m in analysis['model_comparison'] if m['provider'] == 'groq']
    
    print(f"\n🎯 PROVIDER RECOMMENDATIONS:")
    
    if groq_models:
        print(f"\n   ✅ Use Groq for:")
        print(f"      - Speed-critical applications")
        print(f"      - Up to 14,400 requests/day")
        print(f"      - Development and testing")
    
    if ollama_models:
        print(f"\n   ✅ Use Ollama for:")
        print(f"      - Unlimited queries (no API limits)")
        print(f"      - Privacy-sensitive data")
        print(f"      - Offline/on-premise deployment")
        print(f"      - Cost savings at scale")


def save_detailed_results(all_results: Dict, analysis: Dict):
    """
    Save detailed results to files
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results as JSON
    json_filename = f"free_model_test_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_queries': TEST_QUERIES,
            'results': all_results,
            'analysis': analysis
        }, f, indent=2, default=str)
    
    print(f"\n💾 Raw results saved to: {json_filename}")
    
    # Save comparison as CSV
    if analysis.get('model_comparison'):
        df = pd.DataFrame(analysis['model_comparison'])
        csv_filename = f"free_model_comparison_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"💾 Comparison table saved to: {csv_filename}")
    
    # Save individual answers
    answers_filename = f"free_model_answers_{timestamp}.txt"
    with open(answers_filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DETAILED ANSWERS FROM ALL MODELS\n")
        f.write("="*70 + "\n\n")
        
        for query_data in TEST_QUERIES:
            query_id = query_data['id']
            f.write(f"\n{'='*70}\n")
            f.write(f"Query {query_id}: {query_data['query']}\n")
            f.write(f"{'='*70}\n\n")
            
            for pipeline_name, data in all_results.items():
                result = next((r for r in data['results'] if r['query_id'] == query_id), None)
                
                if result and result.get('success'):
                    f.write(f"\n{'-'*70}\n")
                    f.write(f"Model: {data['config']['name']}\n")
                    f.write(f"Time: {result['total_time']:.2f}s | Length: {result['answer_length']} chars\n")
                    f.write(f"{'-'*70}\n\n")
                    f.write(result['answer'])
                    f.write("\n\n")
    
    print(f"💾 Detailed answers saved to: {answers_filename}")


# In[ ]:


def main():
    """
    Main testing function
    """
    print("\n" + "="*70)
    print("🚀 FREE MODEL TESTING SUITE")
    print("Testing all 7 problematic queries across all available free models")
    print("="*70)
    
    # Run tests
    all_results = test_all_models()
    
    if not all_results:
        print("\n❌ No models were tested successfully")
        return
    
    # Analyze results
    analysis = analyze_results(all_results)
    
    # Print recommendations
    print_recommendations(analysis)
    
    # Save results
    save_detailed_results(all_results, analysis)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"   - Models tested: {len(all_results)}")
    print(f"   - Queries per model: {len(TEST_QUERIES)}")
    print(f"   - Total tests: {len(all_results) * len(TEST_QUERIES)}")
    #print("\n💡 Check the saved files for detailed results!")


if __name__ == "__main__":
    main()


# extra_large_minilm (233,027 items) 🏆  
# 
# ✅ BEST overall results when paired with Groq  
# ✅ Most detailed answers (1635 chars avg)  
# ✅ Still very fast with Groq (2.59s)  
# ⚠️ Slowest retrieval (387ms from original test)  

# ## Test Result Visualization

# In[15]:


import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def load_latest_results():
    """Load the most recent test results"""
    results_dir = Path(".")
    result_files = list(results_dir.glob("free_model_test_results_*.json"))
    
    if not result_files:
        print("❌ No test results found!")
        print("   Run: python test_all_free_models.py first")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def create_speed_comparison_chart(analysis_data):
    """Create speed comparison bar chart"""
    models = analysis_data['model_comparison']
    
    if not models:
        return
    
    # Prepare data
    names = [m['name'].replace('🆓 ', '').replace(' (', '\n(') for m in models]
    times = [m['avg_time'] for m in models]
    providers = [m['provider'] for m in models]
    
    # Create color map
    color_map = {
        'ollama': '#00A67E',
        'groq': '#FF6B6B',
        'huggingface': '#FFD93D',
        'local': '#6C5CE7'
    }
    colors = [color_map.get(p, '#95a5a6') for p in providers]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, times, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{time:.2f}s', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('⚡ Speed Comparison: Free LLM Models', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[p], label=p.capitalize()) 
                      for p in set(providers)]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('speed_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: speed_comparison.png")
    plt.close()


def create_quality_comparison_chart(analysis_data):
    """Create quality comparison chart (answer length as proxy)"""
    models = analysis_data['model_comparison']
    
    if not models:
        return
    
    names = [m['name'].replace('🆓 ', '').replace(' (', '\n(') for m in models]
    lengths = [m['avg_answer_length'] for m in models]
    providers = [m['provider'] for m in models]
    
    color_map = {
        'ollama': '#00A67E',
        'groq': '#FF6B6B',
        'huggingface': '#FFD93D',
        'local': '#6C5CE7'
    }
    colors = [color_map.get(p, '#95a5a6') for p in providers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, lengths, color=colors, alpha=0.8)
    
    for bar, length in zip(bars, lengths):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{length:.0f}', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Average Answer Length (characters)', fontsize=12, fontweight='bold')
    ax.set_title('📚 Quality Comparison: Answer Detail Level', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[p], label=p.capitalize()) 
                      for p in set(providers)]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('quality_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: quality_comparison.png")
    plt.close()


def create_scatter_plot(analysis_data):
    """Create speed vs quality scatter plot"""
    models = analysis_data['model_comparison']
    
    if not models:
        return
    
    times = [m['avg_time'] for m in models]
    lengths = [m['avg_answer_length'] for m in models]
    names = [m['name'].replace('🆓 ', '').split(' (')[0] for m in models]
    providers = [m['provider'] for m in models]
    
    color_map = {
        'ollama': '#00A67E',
        'groq': '#FF6B6B',
        'huggingface': '#FFD93D',
        'local': '#6C5CE7'
    }
    colors = [color_map.get(p, '#95a5a6') for p in providers]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(times, lengths, c=colors, s=300, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], lengths[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('⚡ Speed (seconds) - Lower is Better', fontsize=12, fontweight='bold')
    ax.set_ylabel('📚 Detail Level (characters) - Higher is Better', fontsize=12, fontweight='bold')
    ax.set_title('🎯 Speed vs Quality Tradeoff', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    median_time = np.median(times)
    median_length = np.median(lengths)
    ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(median_length, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(0.95, 0.95, 'Slow but Detailed', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, alpha=0.5)
    ax.text(0.05, 0.95, 'Fast and Detailed\n(BEST)', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, alpha=0.5, fontweight='bold')
    ax.text(0.05, 0.05, 'Fast but Brief', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, alpha=0.5)
    ax.text(0.95, 0.05, 'Slow and Brief\n(WORST)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('speed_vs_quality.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: speed_vs_quality.png")
    plt.close()


def create_query_performance_heatmap(all_results):
    """Create heatmap showing performance across queries"""
    
    # Prepare data
    models = list(all_results.keys())
    queries = ["BC Trails", "Concrete QC", "Wheelchair ON", "Bicycle", "Banff", "Alberta Mtns", "Toronto"]
    
    # Create matrix
    time_matrix = []
    for model_name in models:
        model_data = all_results[model_name]
        times = []
        for result in model_data['results']:
            if result.get('success'):
                times.append(result['total_time'])
            else:
                times.append(np.nan)
        time_matrix.append(times)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_labels = [all_results[m]['config']['name'].replace('🆓 ', '').split(' (')[0] for m in models]
    
    im = ax.imshow(time_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(queries)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(queries)
    ax.set_yticklabels(model_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(len(model_labels)):
        for j in range(len(queries)):
            if not np.isnan(time_matrix[i][j]):
                text = ax.text(j, i, f'{time_matrix[i][j]:.1f}s',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('⏱️  Response Time Heatmap (seconds)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Response Time (s)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('query_performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: query_performance_heatmap.png")
    plt.close()


def create_success_rate_chart(analysis_data):
    """Create success rate comparison"""
    models = analysis_data['model_comparison']
    
    if not models:
        return
    
    names = [m['name'].replace('🆓 ', '').replace(' (', '\n(') for m in models]
    success_rates = [m['success_rate'] for m in models]
    providers = [m['provider'] for m in models]
    
    color_map = {
        'ollama': '#00A67E',
        'groq': '#FF6B6B',
        'huggingface': '#FFD93D',
        'local': '#6C5CE7'
    }
    colors = [color_map.get(p, '#95a5a6') for p in providers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, success_rates, color=colors, alpha=0.8)
    
    for bar, rate in zip(bars, success_rates):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{rate:.0f}%', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_title('✅ Success Rate: Query Completion', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('success_rate.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: success_rate.png")
    plt.close()


def create_comprehensive_comparison_table(analysis_data):
    """Create a comprehensive comparison table image"""
    models = analysis_data['model_comparison']
    
    if not models:
        return
    
    # Prepare data
    data = []
    for m in models:
        data.append([
            m['name'].replace('🆓 ', '').split(' (')[0],
            m['provider'].capitalize(),
            f"{m['avg_time']:.2f}s",
            f"{m['avg_answer_length']:.0f}",
            f"{m['success_rate']:.0f}%",
            "Fast" if m['avg_time'] < 3 else "Medium" if m['avg_time'] < 10 else "Slow"
        ])
    
    fig, ax = plt.subplots(figsize=(14, len(models) + 2))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Model', 'Provider', 'Avg Time', 'Avg Length', 'Success', 'Rating']
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('📊 Comprehensive Model Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: comparison_table.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATION CHARTS")
    print("="*70 + "\n")
    
    # Load data
    data = load_latest_results()
    if not data:
        return
    
    analysis = data.get('analysis', {})
    all_results = data.get('results', {})
    
    if not analysis or not all_results:
        print("❌ Invalid data structure")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    print("📈 Creating charts...")
    
    # Generate all charts
    try:
        create_speed_comparison_chart(analysis)
        create_quality_comparison_chart(analysis)
        create_scatter_plot(analysis)
        create_query_performance_heatmap(all_results)
        create_success_rate_chart(analysis)
        create_comprehensive_comparison_table(analysis)
    except Exception as e:
        print(f"⚠️  Error creating chart: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ ALL CHARTS GENERATED!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 speed_comparison.png")
    print("  📊 quality_comparison.png")
    print("  📊 speed_vs_quality.png")
    print("  📊 query_performance_heatmap.png")
    print("  📊 success_rate.png")
    print("  📊 comparison_table.png")


if __name__ == "__main__":
    main()

