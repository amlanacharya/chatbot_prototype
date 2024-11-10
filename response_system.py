from step1 import AimagineKnowledgeBase
from web_search import WebSearcher
from groq_integration import GroqGenerator
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Response:
    content: str
    confidence: float
    source: str
    metadata: Dict = None

class ResponseSystem:
    def __init__(self, kb: AimagineKnowledgeBase):
        self.kb = kb
        self.web_searcher = WebSearcher()
        self.groq = GroqGenerator()
        self.confidence_thresholds = {
            'kb': 0.6,
            'rag': 0.5,
            'groq': 0.4
        }

    async def _get_kb_response(self, query: str) -> Optional[Response]:
        """Get response from Knowledge Base"""
        try:
            results = self.kb.search_similar_content(query)
            if not results:
                return None
                
            # Get the best result
            best_result = max(results, key=lambda x: x['similarity'])
            
            return Response(
                content=best_result['content'],
                confidence=best_result['similarity'],
                source='knowledge_base',
                metadata={
                    'type': best_result['type'],
                    'raw_response': best_result
                }
            )
        except Exception as e:
            print(f"Error in KB response: {e}")
            return None

    async def get_response(self, query: str) -> Response:
        """Multi-tiered response system"""
        # Tier 1: Knowledge Base
        kb_response = await self._get_kb_response(query)
        if kb_response and kb_response.confidence >= self.confidence_thresholds['kb']:
            return kb_response
            
        # Tier 2: RAG
        rag_response = await self._get_rag_response(query, kb_response)
        if rag_response and rag_response.confidence >= self.confidence_thresholds['rag']:
            return rag_response
            
        # Tier 3: Groq
        groq_response = await self._get_groq_response(query, kb_response, rag_response)
        if groq_response and groq_response.confidence >= self.confidence_thresholds['groq']:
            return groq_response
            
        # Return best available response
        responses = [r for r in [kb_response, rag_response, groq_response] if r]
        if responses:
            return max(responses, key=lambda x: x.confidence)
            
        # Escalate if no good response
        return self._escalate_to_human(query)

    async def _get_rag_response(self, query: str, kb_response: Optional[Response]) -> Optional[Response]:
        """Get response using RAG"""
        try:
            # Get relevant documents from web
            search_results = await self.web_searcher.search(query)
            
            if not search_results:
                return None
                
            # Get the most relevant result
            best_result = max(search_results, 
                            key=lambda x: self._calculate_relevance_score(query, x['snippet']))
            
            # Calculate confidence
            confidence = self._calculate_relevance_score(query, best_result['snippet'])
            
            # Format response
            response_text = self._format_rag_response(best_result, kb_response)
            
            return Response(
                content=response_text,
                confidence=confidence,
                source='rag',
                metadata={
                    'url': best_result['url'],
                    'title': best_result['title'],
                    'kb_response': kb_response.content if kb_response else None
                }
            )
            
        except Exception as e:
            print(f"Error in RAG response: {e}")
            return None

    def _format_rag_response(self, search_result: Dict, kb_response: Optional[Response]) -> str:
        """Format RAG response combining web search and KB results"""
        response_parts = []
        
        # Add main response from web search
        response_parts.append(f"{search_result['snippet']}")
        
        # Add source attribution
        response_parts.append(f"\nSource: {search_result['title']}")
        
        # Add KB information if available
        if kb_response:
            response_parts.append("\nAdditional information from our knowledge base:")
            response_parts.append(kb_response.content)
            
        return "\n".join(response_parts)

    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        try:
            query_embedding = self.kb.create_embedding(query)
            content_embedding = self.kb.create_embedding(content)
            return float(self.kb.cosine_similarity(query_embedding, content_embedding))
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.0

    def _escalate_to_human(self, query: str) -> Response:
        """Handle escalation to human agent"""
        return Response(
            content="I apologize, but I'll need to transfer you to a human agent for better assistance. "
                   "Please hold while I connect you.",
            confidence=1.0,
            source='escalation',
            metadata={'original_query': query}
        )

    async def _get_groq_response(
        self, 
        query: str, 
        kb_response: Optional[Response],
        rag_response: Optional[Response]
    ) -> Optional[Response]:
        """Generate response using Groq"""
        try:
            # Prepare context from previous responses
            context = self._prepare_groq_context(kb_response, rag_response)
            
            # Generate response using Groq
            groq_result = self.groq.generate_response(query, context)
            
            return Response(
                content=groq_result["text"],
                confidence=groq_result["confidence"],
                source='groq',
                metadata={
                    'kb_response': kb_response.content if kb_response else None,
                    'rag_response': rag_response.content if rag_response else None
                }
            )
            
        except Exception as e:
            print(f"Error in Groq response: {e}")
            return None

    def _prepare_groq_context(
        self,
        kb_response: Optional[Response],
        rag_response: Optional[Response]
    ) -> Optional[str]:
        """Prepare context for Groq from previous responses"""
        context_parts = []
        
        if kb_response:
            context_parts.append(f"From our knowledge base: {kb_response.content}")
            
        if rag_response:
            context_parts.append(f"From web search: {rag_response.content}")
            
        return "\n\n".join(context_parts) if context_parts else None

# Test the response system
async def test_response_system():
    try:
        # Initialize knowledge base and response system
        kb = AimagineKnowledgeBase()
        response_system = ResponseSystem(kb)
        
        # Test queries
        test_queries = [
            "What is the baggage allowance?",
            "Can I bring my pet on the flight?",
            "What's the status of flight AI123?",
        ]
        
        print("\nTesting Response System:")
        print("------------------------")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await response_system.get_response(query)
            print(f"Response: {response.content}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Source: {response.source}")
            
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'kb' in locals():
            kb.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_response_system()) 