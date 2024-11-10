import aiohttp
import asyncio
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import os

class WebSearcher:
    def __init__(self):
        self.base_url = "https://duckduckgo.com/html/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup log file path
        log_file = os.path.join(log_dir, f'web_search_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        try:
            self.logger.info(f"Searching for: {query}")
            
            # Add airline-specific keywords to query
            enhanced_query = f"{query} site:aimagine-airlines.com OR airline OR flight OR travel"
            
            async with aiohttp.ClientSession() as session:
                # Make the search request
                async with session.get(
                    self.base_url,
                    params={'q': enhanced_query},
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Search failed with status {response.status}")
                        return []
                        
                    html = await response.text()
                    
            # Parse results
            results = await self._parse_results(html, max_results)
            self.logger.info(f"Found {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    async def _parse_results(self, html: str, max_results: int) -> List[Dict]:
        """Parse HTML and extract search results"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Find all result containers
        for result in soup.select('.result')[:max_results]:
            try:
                # Extract title
                title_elem = result.select_one('.result__title')
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                # Extract URL
                url_elem = result.select_one('.result__url')
                url = url_elem.get_text(strip=True) if url_elem else ''
                
                # Extract snippet
                snippet_elem = result.select_one('.result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                if title and snippet:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'web_search'
                    })
                    
            except Exception as e:
                self.logger.error(f"Error parsing result: {str(e)}")
                continue
                
        return results

    async def get_relevant_content(self, query: str) -> Optional[Dict]:
        """
        Get the most relevant content for a query
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary containing most relevant result or None
        """
        results = await self.search(query)
        
        if not results:
            return None
            
        # Get most relevant result (first result)
        best_result = results[0]
        
        return {
            'content': best_result['snippet'],
            'source_url': best_result['url'],
            'title': best_result['title']
        }

# Example usage and testing
async def test_web_search():
    searcher = WebSearcher()
    
    # Test queries
    test_queries = [
        "airline baggage allowance",
        "flight cancellation policy",
        "airline pet travel rules"
    ]
    
    print("\nTesting Web Search:")
    print("-----------------")
    
    for query in test_queries:
        print(f"\nSearching for: {query}")
        results = await searcher.search(query, max_results=2)
        
        if results:
            print("\nResults found:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"URL: {result['url']}")
                print(f"Snippet: {result['snippet'][:200]}...")
        else:
            print("No results found")

if __name__ == "__main__":
    asyncio.run(test_web_search()) 