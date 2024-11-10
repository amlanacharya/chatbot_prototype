from step1 import AimagineKnowledgeBase

def setup():
    try:
        print("Initializing knowledge base...")
        kb = AimagineKnowledgeBase()
        
        print("Clearing database...")
        kb.clear_database()
        
        print("Importing knowledge base...")
        kb.import_knowledge_base('knowledge_base.txt')
        
        print("Setup complete!")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'kb' in locals():
            kb.close()

if __name__ == "__main__":
    setup() 