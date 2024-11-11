import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st
from response_system import ResponseSystem
from step1 import AimagineKnowledgeBase
import asyncio
from datetime import datetime
import json
import os


class ChatInterface:
    def __init__(self):
        self.kb = AimagineKnowledgeBase()
        self.response_system = ResponseSystem(self.kb)
        
    def initialize_session(self):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Initialize conversation log
        if "conversation_log" not in st.session_state:
            st.session_state.conversation_log = []

    def save_conversation(self):
        """Save conversation to session state instead of file system"""
        if st.session_state.conversation_log:
            # Store in session state instead of file system
            if "saved_conversations" not in st.session_state:
                st.session_state.saved_conversations = []
            
            st.session_state.saved_conversations.append({
                "timestamp": datetime.now().isoformat(),
                "conversation": st.session_state.conversation_log
            })

    def display_chat_history(self):
        """Display chat messages"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    async def get_assistant_response(self, user_input: str):
        """Get response from the response system"""
        response = await self.response_system.get_response(user_input)
        return response

    def log_interaction(self, user_input: str, response):
        """Log the interaction details"""
        st.session_state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": {
                "content": response.content,
                "source": response.source,
                "confidence": response.confidence,
                "metadata": response.metadata
            }
        })

    def run(self):
        try:
            # Page config
            st.set_page_config(
                page_title="Aimagine Airlines Assistant",
                page_icon="✈️"
            )
            
            # Header
            st.header("✈️ Aimagine Airlines Assistant")
            
            # Initialize session
            self.initialize_session()
            
            # Display chat history
            self.display_chat_history()
            
            # Chat input
            if user_input := st.chat_input("How can I help you today?"):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                    
                # Get assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = asyncio.run(self.get_assistant_response(user_input))
                        st.markdown(response.content)
                        
                        # Show source if not from knowledge base
                        if response.source != 'knowledge_base':
                            st.caption(f"Source: {response.source}")
                            
                        # Show confidence score in sidebar
                        with st.sidebar:
                            st.metric("Response Confidence", f"{response.confidence:.2%}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                # Log interaction
                self.log_interaction(user_input, response)
                
                # Save conversation
                self.save_conversation()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Cleanup
            if hasattr(self, 'kb'):
                self.kb.close()

def main():
    chat_interface = ChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main() 