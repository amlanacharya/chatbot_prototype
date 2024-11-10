from groq import Groq
from typing import Dict, Optional
import logging
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class GroqGenerator:
    def __init__(self, model_name="llama3-8b-8192"):
        self.setup_logging()
        self.model = model_name
        
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        self.client = Groq(api_key=api_key)
        self.logger.info("Groq client initialized successfully")

    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join(log_dir, 'groq.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_response(
        self, 
        query: str, 
        context: Optional[str] = None,
    ) -> Dict:
        """
        Generate a response using Groq API
        
        Args:
            query: User query
            context: Additional context (e.g., from KB or RAG)
            
        Returns:
            Dict containing response text and confidence score
        """
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self._prepare_system_prompt(context)
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            # Generate response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.48,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None
            )
            
            # Collect response
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            # Calculate confidence
            confidence = self._calculate_confidence(response_text, query)
            
            return {
                "text": response_text,
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "text": "I apologize, but I'm having trouble generating a response.",
                "confidence": 0.0
            }

    def _prepare_system_prompt(self, context: Optional[str] = None) -> str:
        """Prepare system prompt with airline context"""
        base_prompt = """You are an AI assistant for Aimagine Airlines, a premium airline company. 
        Your role is to provide helpful, accurate, and professional responses to customer queries. 
        Always maintain a friendly and empathetic tone.
        """
        
        if context:
            base_prompt += f"\n\nAdditional Context: {context}"
            
        return base_prompt

    def _calculate_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score for the generated response"""
        confidence = 0.7  # Base confidence
        
        # Length check
        if len(response) < 20:
            confidence -= 0.2
        elif len(response) > 500:
            confidence -= 0.1
            
        # Keyword check for airline-related terms
        airline_keywords = {
            'flight', 'airline', 'travel', 'passenger', 'airport',
            'baggage', 'ticket', 'booking', 'reservation', 'schedule'
        }
        
        response_words = set(response.lower().split())
        keyword_matches = response_words.intersection(airline_keywords)
        
        confidence += min(len(keyword_matches) * 0.05, 0.2)
        
        return min(max(confidence, 0.0), 1.0) 