"""
Tutor Agent - MI Knowledge Hub
Interactive learning support and Q&A
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class TutorAgent(BaseAgent):
    """Tutor Agent for interactive learning support."""

    def __init__(self):
        super().__init__(
            name="Tutor Agent",
            description="Interactive learning support agent",
            temperature=0.8,
        )

    async def execute(self, question: str, **kwargs) -> Dict[str, Any]:
        """Answer user questions interactively."""
        self.logger.info(f"Answering question: {question[:50]}...")

        try:
            answer = await self.generate_response(question)
            return {"success": True, "question": question, "answer": answer}
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return {"success": False, "error": str(e)}
