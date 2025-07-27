# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
"""
Python MCP Web Scraper Server and Ollama Client Example

This module demonstrates:
- An MCP server exposing a web scraping tool using FastMCP (with agent card, JRPC, SSE support)
- An Ollama-compatible MCP client that calls the scraper tool

Dependencies:
    pip install requests beautifulsoup4 fastmcp ollama

Usage:
    # Start the server:
    python p4.py server

    # Run the client:
    python p4.py client --url https://example.com
"""

import sys
import multiprocessing

# --- MCP Server Implementation ---
def run_server():
    from fastmcp import MCPServer
    import requests
    from bs4 import BeautifulSoup

    AGENT_CARD = {
        "id": "webscraper-agent-1",
        "name": "WebScraperAgent",
        "description": "Scrapes web pages and returns clean text content.",
        "protocols": ["mcp", "jrpc", "sse"],
        "tools": [
            {
                "name": "scrape",
                "description": "Scrape a web page and return its text content.",
                "input_schema": {"url": "string"},
                "output_schema": {"content": "string"}
            }
        ]
    }

    class WebScraperServer(MCPServer):
        def scrape(self, url: str) -> str:
            """Scrapes the given URL and returns text content."""
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)

    server = WebScraperServer(agent_card=AGENT_CARD)
    server.register_tool("scrape", server.scrape)

    @server.route("/agent-card")
    def agent_card_endpoint():
        return AGENT_CARD

    print("Starting MCP WebScraperServer on http://0.0.0.0:5000 ...")
    server.run(host="0.0.0.0", port=5000)

# --- Ollama MCP Client Implementation ---
def run_client(url: str):
    import requests

    class MCPClient:
        def __init__(self, server_url):
            self.server_url = server_url

        def call_tool(self, tool_name, url):
            # For this example, we assume the server exposes a JRPC endpoint at /jrpc
            payload = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": {"url": url},
                "id": 1
            }
            resp = requests.post(f"{self.server_url}/jrpc", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                # If result is a dict with 'content', return it
                if isinstance(data["result"], dict) and "content" in data["result"]:
                    return data["result"]["content"]
                return data["result"]
            elif "error" in data:
                raise Exception(f"Server error: {data['error']}")
            else:
                raise Exception("Unexpected response from server")

    client = MCPClient(server_url="http://localhost:5000")
    print(f"Requesting scrape for: {url}")
    response = client.call_tool("scrape", url)
    print("Scraped Content:\n", response)

def run_both(url: str):
    """Run both MCP server and client in separate processes."""
    server_proc = multiprocessing.Process(target=run_server)
    client_proc = multiprocessing.Process(target=run_client, args=(url,))
    server_proc.start()
    # Wait a bit for server to start before client requests
    import time
    time.sleep(2)
    client_proc.start()
    client_proc.join()
    server_proc.terminate()

# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Web Scraper Server/Client Example")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server", help="Run the MCP web scraper server")
    client_parser = subparsers.add_parser("client", help="Run the Ollama MCP client")
    client_parser.add_argument("--url", required=True, help="URL to scrape")
    both_parser = subparsers.add_parser("both", help="Run both server and client (for demo)")
    both_parser.add_argument("--url", required=True, help="URL to scrape")

    args = parser.parse_args()

    if args.mode == "server":
        run_server()
    elif args.mode == "client":
        run_client(args.url)
    elif args.mode == "both":
        run_both(args.url)
