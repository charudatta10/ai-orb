# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
import logging
import ollama  # Using Ollama as the local LLM server

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class AI_Agent:
    """Generic AI Agent that loads models via Ollama LLM server."""
    
    def __init__(self, name: str, model: str, tool: str = None):
        self.name = name
        self.model = model
        self.tool = tool
    
    def execute(self, prompt: str, context: str = "") -> str:
        """Executes a task using Ollama with the specified model."""
        full_prompt = f"""
        Context:
        {context}

        Task:
        {prompt}

        Requirements:
        - Well-structured Markdown format
        - Technical accuracy
        - High-quality refinement
        """

        try:
            result = ollama.generate(model=self.model, prompt=full_prompt, stream=False)
            return result.get("content", "No response generated.")
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"# Generation Error\n\n{str(e)}"

class MultiAgentSystem:
    """Manages collaboration among AI agents using Ollama."""

    def __init__(self):
        self.agents = {
            "web_scraper": AI_Agent("WebScraper", "deepseek-r1:7b", tool="web-scraping"),
            "pdf_reader": AI_Agent("PDFReader", "deepseek-r1:7b", tool="pdf-processing"),
            "editor": AI_Agent("Editor", "llama3.2:3b", tool="memory-enhanced-editing"),
            "reviewer": AI_Agent("Reviewer", "llama3.2:3b", tool="critical-evaluation")
        }

    def generate_article(self, topic, pdf_source):
        iteration = 1
        approved = False

        while not approved:
            logging.info(f"==== Iteration {iteration} ====")

            # Step 1: Web Scraper Draft
            web_draft = self.agents["web_scraper"].execute(f"Scrape data about {topic}")

            # Step 2: PDF Reader Draft
            pdf_draft = self.agents["pdf_reader"].execute(f"Extract insights from {pdf_source}")

            # Step 3: Editor Merges Drafts
            final_draft = self.agents["editor"].execute(f"Merge drafts:\nWeb:\n{web_draft}\nPDF:\n{pdf_draft}")

            # Step 4: Reviewer Scores Draft
            score = self.agents["reviewer"].execute(f"Review and score:\n{final_draft}")

            # Every 10th iteration sends draft for human evaluation
            if iteration % 10 == 0:
                logging.info("Sending draft for human evaluation.")

            if int(score) >= 8:
                logging.info("✅ Draft approved!")
                approved = True
            else:
                logging.info("🔄 Refining the draft...")

            iteration += 1

        return final_draft

# Run the system
if __name__ == "__main__":
    system = MultiAgentSystem()
    final_article = system.generate_article("Meditation: A Way of Life", "meditation.pdf")
    logging.info(f"\nFinal Article:\n{final_article}")