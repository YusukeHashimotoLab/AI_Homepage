"""
Content Agent - MI Knowledge Hub
Generates and updates educational articles
"""

from typing import Dict, Any
from pathlib import Path

from .base_agent import BaseAgent


class ContentAgent(BaseAgent):
    """
    Content Agent for article generation and updates.

    Responsibilities:
    - Generate educational articles
    - Update existing content
    - Follow 9-phase quality workflow
    - Integrate with Academic Reviewer
    """

    def __init__(self):
        super().__init__(
            name="Content Agent",
            description="Article generation and content management agent",
            temperature=0.7,
        )

    async def execute(
        self,
        topic: str,
        level: str = "intermediate",
        target_audience: str = "undergraduate",
        output_file: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate educational article.

        Args:
            topic: Article topic
            level: Difficulty level (beginner/intermediate/advanced)
            target_audience: Target audience description
            output_file: Output file path

        Returns:
            Dictionary with generation results
        """
        self.logger.info(f"Generating article on '{topic}' (level: {level})")

        try:
            # Generate article content
            article = await self._generate_article(topic, level, target_audience)

            # Save to file if specified
            if output_file:
                output_path = self.content_dir / output_file
                self.save_text(article, output_path)

            return {
                "success": True,
                "topic": topic,
                "level": level,
                "output_file": str(output_path) if output_file else None,
                "word_count": len(article.split()),
            }

        except Exception as e:
            self.logger.error(f"Error in execute: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_article(
        self, topic: str, level: str, target_audience: str
    ) -> str:
        """
        Generate article content using Claude.

        Args:
            topic: Article topic
            level: Difficulty level
            target_audience: Target audience

        Returns:
            Generated article markdown
        """
        prompt = f"""Generate an educational article about {topic}.

Level: {level}
Target Audience: {target_audience}

The article should:
- Start with clear learning objectives
- Include relevant theory and concepts
- Provide practical examples
- Include references to recent research
- Be written in Japanese
- Use markdown format

Please generate a comprehensive article."""

        article = await self.generate_response(prompt)
        return article


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = ContentAgent()
        result = await agent.execute(
            topic="ベイズ最適化", level="intermediate", output_file="test_article.md"
        )
        print(f"Result: {result}")

    asyncio.run(main())
