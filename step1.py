from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from typing import List, Dict
import textwrap
import streamlit as st

# Aura DB Credentials
NEO4J_URI = "neo4j+s://791e58dc.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "mY-OjrlFe-gpDj8wewBwbPO6J4IiuHKgCjn5W03eIJ4"

class AimagineKnowledgeBase:
    def __init__(self):
        # Use Streamlit secrets instead of environment variables
        self.driver = GraphDatabase.driver(
            st.secrets["NEO4J_URI"],
            auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.dimension = 384  # MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize document chunks storage
        self.chunks: List[Dict] = []
        
        # Test connection
        try:
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j!")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise

    def close(self):
        self.driver.close()

    def create_embedding(self, text):
        # Return numpy array directly
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def import_knowledge_base(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        with self.driver.session() as session:
            # Create main categories
            session.run("""
                CREATE (faqs:Category {name: 'FAQs'})
                CREATE (policies:Category {name: 'Policies'})
                CREATE (other:Category {name: 'Other Documents'})
            """)

            # Import FAQs
            self._import_faqs(session, content)
            
            # Import Policies
            self._import_policies(session, content)
            
            # Import Other Documents
            self._import_other_documents(session, content)

    def _import_faqs(self, session, content):
        faq_pattern = r'1\. FAQs(.*?)2\. Policies'
        faq_match = re.search(faq_pattern, content, re.DOTALL)
        
        if faq_match:
            faq_content = faq_match.group(1)
            faq_items = re.findall(r'(\d+\.\d+\.?\d*\.) (.*?)\n- (.*?)(?=\n\d|\Z)', faq_content, re.DOTALL)
            
            for item in faq_items:
                question = item[1].strip()
                answer = item[2].strip()
                
                # Create embedding and convert numpy array to list for Neo4j storage
                embedding = self.create_embedding(question + " " + answer)
                embedding_list = embedding.tolist()  # Convert numpy array to list
                
                session.run("""
                    MATCH (c:Category {name: 'FAQs'})
                    CREATE (f:FAQ {
                        question: $question,
                        answer: $answer,
                        embedding: $embedding
                    })
                    CREATE (c)-[:CONTAINS]->(f)
                """, question=question, answer=answer, embedding=embedding_list)

    def _import_policies(self, session, content):
        # Extract Policies section
        policy_pattern = r'2\. Policies(.*?)3\. Other relevant documents'
        policy_match = re.search(policy_pattern, content, re.DOTALL)
        
        if policy_match:
            policy_content = policy_match.group(1)
            # Extract individual policies
            policy_items = re.findall(r'(\d+\.\d+\.?\d*\.) (.*?)\n- (.*?)(?=\n\d|\Z)', policy_content, re.DOTALL)
            
            for item in policy_items:
                name = item[1].strip()
                description = item[2].strip()
                
                # Create embedding and convert to list for Neo4j storage
                embedding = self.create_embedding(name + " " + description)
                embedding_list = embedding.tolist()
                
                session.run("""
                    MATCH (c:Category {name: 'Policies'})
                    CREATE (p:Policy {
                        name: $name,
                        description: $description,
                        embedding: $embedding
                    })
                    CREATE (c)-[:CONTAINS]->(p)
                """, name=name, description=description, embedding=embedding_list)

    def _import_other_documents(self, session, content):
        sections = {
            'AllowedItems': self._create_allowed_items_node,
            'HolidayOffers': self._create_holiday_offers_node,
            'CrossSelling': self._create_cross_selling_node,
            'BusinessClass': self._create_business_class_node
        }

        for section_name, create_function in sections.items():
            create_function(session, content)

    def _create_allowed_items_node(self, session, content):
        # Extract Allowed Items section
        allowed_items_pattern = r'3\.10\. Allowed and Not Allowed Items(.*?)3\.11\.'
        match = re.search(allowed_items_pattern, content, re.DOTALL)
        
        if match:
            items_content = match.group(1)
            session.run("""
                MATCH (c:Category {name: 'Other Documents'})
                CREATE (a:AllowedItems {
                    content: $content,
                    embedding: $embedding
                })
                CREATE (c)-[:CONTAINS]->(a)
            """, content=items_content.strip(), embedding=self.create_embedding(items_content))

    def _create_holiday_offers_node(self, session, content):
        # Extract Holiday Offers section
        holiday_pattern = r'3\.11\. Holiday Offers(.*?)3\.12\.'
        match = re.search(holiday_pattern, content, re.DOTALL)
        
        if match:
            offers_content = match.group(1)
            session.run("""
                MATCH (c:Category {name: 'Other Documents'})
                CREATE (h:HolidayOffers {
                    content: $content,
                    embedding: $embedding
                })
                CREATE (c)-[:CONTAINS]->(h)
            """, content=offers_content.strip(), embedding=self.create_embedding(offers_content))

    def _create_cross_selling_node(self, session, content):
        # Extract Cross-Selling section
        cross_selling_pattern = r'3\.12\. Cross-Selling Opportunities(.*?)3\.13\.'
        match = re.search(cross_selling_pattern, content, re.DOTALL)
        
        if match:
            cross_selling_content = match.group(1)
            session.run("""
                MATCH (c:Category {name: 'Other Documents'})
                CREATE (cs:CrossSelling {
                    content: $content,
                    embedding: $embedding
                })
                CREATE (c)-[:CONTAINS]->(cs)
            """, content=cross_selling_content.strip(), embedding=self.create_embedding(cross_selling_content))

    def _create_business_class_node(self, session, content):
        # Extract Business Class section
        business_pattern = r'3\.13\. Business Class Perks and Privileges(.*?)$'
        match = re.search(business_pattern, content, re.DOTALL)
        
        if match:
            business_content = match.group(1)
            session.run("""
                MATCH (c:Category {name: 'Other Documents'})
                CREATE (b:BusinessClass {
                    content: $content,
                    embedding: $embedding
                })
                CREATE (c)-[:CONTAINS]->(b)
            """, content=business_content.strip(), embedding=self.create_embedding(business_content))

    def search_similar_content(self, query, k=3, threshold=0.7):
        # Create query embedding as numpy array
        query_embedding = self.create_embedding(query)
        
        with self.driver.session() as session:
            # Get all embeddings from Neo4j
            result = session.run("""
                MATCH (n)
                WHERE (n:FAQ OR n:Policy)
                RETURN n, labels(n) as labels, n.embedding as embedding
            """)
            
            # Prepare FAISS index
            index = faiss.IndexFlatL2(self.dimension)
            nodes = []
            embeddings = []
            
            # Collect results
            for record in result:
                node_data = record.get("n")  # Get node properties
                if node_data and "embedding" in node_data:  # Check if node and embedding exist
                    nodes.append({
                        'properties': node_data,
                        'labels': record.get("labels", [])
                    })
                    # Convert stored list back to numpy array
                    embedding = np.array(node_data["embedding"]).astype('float32')
                    embeddings.append(embedding)
            
            if not embeddings:
                return None
                
            # Stack embeddings into a single numpy array
            embeddings_array = np.stack(embeddings).astype('float32')
            index.add(embeddings_array)
            
            # Search using FAISS
            D, I = index.search(query_embedding.reshape(1, -1), min(k, len(embeddings)))
            
            # Process results
            responses = []
            for idx, distance in zip(I[0], D[0]):
                if idx < len(nodes) and distance < threshold:  # Add bounds check
                    node_info = nodes[idx]
                    node_props = node_info['properties']
                    
                    # Format response based on node type
                    if 'FAQ' in node_info['labels']:
                        responses.append({
                            'type': 'FAQ',
                            'content': f"Q: {node_props['question']}\nA: {node_props['answer']}",
                            'similarity': float(1 - distance/2)
                        })
                    elif 'Policy' in node_info['labels']:
                        responses.append({
                            'type': 'Policy',
                            'content': f"{node_props['name']}: {node_props['description']}",
                            'similarity': float(1 - distance/2)
                        })
            
            return responses if responses else None

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better processing"""
        chunks = []
        sentences = text.split('.')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size:
                # Store current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_tokens = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_tokens + [sentence]
                current_length = sum(len(t) for t in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# Usage example:
if __name__ == "__main__":
    try:
        # Initialize knowledge base
        kb = AimagineKnowledgeBase()
        
        # Clear existing data
        print("Clearing existing database...")
        kb.clear_database()
        
        # Import knowledge base from file
        print("Importing knowledge base...")
        kb.import_knowledge_base('Aimagine_Airlines/knowledge_base.txt')
        
        # Test some queries
        print("\nTesting queries...")
        queries = [
            "What is the baggage allowance?",
            "Tell me about business class benefits",
            "What are the Christmas offers?",
            "What items are not allowed in hand baggage?"
        ]
        
        for query in queries:
            result = kb.search_similar_content(query)
            if result:
                print(f"\nQuery: {query}")
                print(f"Type: {result['type']}")
                print(f"Content: {result['content'][:200]}...")
                print(f"Similarity: {result['similarity']:.2f}")
            else:
                print(f"\nNo matching result found for query: {query}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Always close the connection
        if 'kb' in locals():
            kb.close()
            print("\nConnection closed.")
