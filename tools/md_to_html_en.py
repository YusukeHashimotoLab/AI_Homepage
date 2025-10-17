#!/usr/bin/env python3
"""
English Markdown to HTML Converter for MI Knowledge Hub

Converts English Markdown files with YAML front matter to standalone HTML pages.
Supports Mermaid diagrams, code syntax highlighting, and responsive design.
"""

import re
import markdown
from pathlib import Path
from typing import Dict, Any
import yaml


def parse_front_matter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML front matter and markdown content."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            front_matter = yaml.safe_load(parts[1])
            markdown_content = parts[2].strip()
            return front_matter, markdown_content
    return {}, content


def create_html_template(title: str, body: str, metadata: Dict[str, Any]) -> str:
    """Create complete HTML page with styling and scripts."""

    reading_time = metadata.get('total_reading_time', metadata.get('reading_time', metadata.get('estimated_time', 'Unknown')))
    level = metadata.get('level', metadata.get('difficulty', ''))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{metadata.get('subtitle', metadata.get('title', ''))}">
    <title>{title} - MI Knowledge Hub</title>

    <!-- CSS Styling -->
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --bg-color: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
            --code-bg: #f5f5f5;
            --link-color: #3498db;
            --link-hover: #2980b9;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            color: var(--text-color);
            background: var(--bg-color);
            padding: 0;
            margin: 0;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}

        /* Header */
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        header .container {{
            padding: 0 1.5rem;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}

        .meta {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            font-size: 0.9rem;
            opacity: 0.95;
            margin-top: 1rem;
        }}

        .meta span {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }}

        /* Typography */
        h2 {{
            font-size: 1.75rem;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid var(--secondary-color);
            color: var(--primary-color);
        }}

        h3 {{
            font-size: 1.4rem;
            margin-top: 2rem;
            margin-bottom: 0.8rem;
            color: var(--primary-color);
        }}

        h4 {{
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 0.6rem;
            color: var(--primary-color);
        }}

        p {{
            margin-bottom: 1.2rem;
        }}

        a {{
            color: var(--link-color);
            text-decoration: none;
            transition: color 0.2s;
        }}

        a:hover {{
            color: var(--link-hover);
            text-decoration: underline;
        }}

        /* Lists */
        ul, ol {{
            margin-left: 2rem;
            margin-bottom: 1.2rem;
        }}

        li {{
            margin-bottom: 0.5rem;
        }}

        /* Code blocks */
        code {{
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}

        pre {{
            background: var(--code-bg);
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
        }}

        pre code {{
            background: none;
            padding: 0;
            font-size: 0.9rem;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1.5rem;
            overflow-x: auto;
            display: block;
        }}

        thead {{
            display: table;
            width: 100%;
            table-layout: fixed;
        }}

        tbody {{
            display: table;
            width: 100%;
            table-layout: fixed;
        }}

        th, td {{
            padding: 0.8rem;
            text-align: left;
            border: 1px solid var(--border-color);
        }}

        th {{
            background: var(--primary-color);
            color: white;
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: #f9f9f9;
        }}

        /* Blockquotes */
        blockquote {{
            border-left: 4px solid var(--secondary-color);
            padding-left: 1.5rem;
            margin: 1.5rem 0;
            font-style: italic;
            color: #666;
        }}

        /* Images */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 1rem 0;
        }}

        /* Mermaid diagrams */
        .mermaid {{
            text-align: center;
            margin: 2rem 0;
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        /* Details/Summary (for exercises) */
        details {{
            margin: 1rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--primary-color);
            padding: 0.5rem;
        }}

        summary:hover {{
            color: var(--secondary-color);
        }}

        /* Footer */
        footer {{
            margin-top: 4rem;
            padding: 2rem 0;
            border-top: 2px solid var(--border-color);
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }}

        /* Navigation buttons */
        .nav-buttons {{
            display: flex;
            justify-content: space-between;
            margin: 3rem 0;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .nav-button {{
            display: inline-block;
            padding: 0.8rem 1.5rem;
            background: var(--secondary-color);
            color: white;
            border-radius: 6px;
            text-decoration: none;
            transition: all 0.3s;
            font-weight: 600;
        }}

        .nav-button:hover {{
            background: var(--link-hover);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            h1 {{
                font-size: 1.6rem;
            }}

            h2 {{
                font-size: 1.4rem;
            }}

            .meta {{
                font-size: 0.85rem;
            }}

            pre {{
                padding: 1rem;
                font-size: 0.85rem;
            }}

            table {{
                font-size: 0.9rem;
            }}
        }}
    </style>

    <!-- Mermaid for diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</head>
<body>
    <header>
        <div class="container">
            <h1>{title}</h1>
            <div class="meta">
                <span>📖 Reading Time: {reading_time}</span>
                <span>📊 Level: {level}</span>
            </div>
        </div>
    </header>

    <main class="container">
        {body}

        <div class="nav-buttons">
            <a href="index.html" class="nav-button">← Back to Series Index</a>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; 2025 MI Knowledge Hub - Dr. Yusuke Hashimoto, Tohoku University</p>
            <p>Licensed under CC BY 4.0</p>
        </div>
    </footer>
</body>
</html>
"""
    return html


def convert_markdown_to_html(md_file: Path, output_dir: Path) -> None:
    """Convert a single markdown file to HTML."""

    # Read markdown file
    content = md_file.read_text(encoding='utf-8')

    # Parse front matter
    metadata, md_content = parse_front_matter(content)

    # Configure markdown with extensions
    md = markdown.Markdown(extensions=[
        'extra',           # Tables, fenced code blocks, etc.
        'codehilite',      # Syntax highlighting
        'toc',             # Table of contents
        'nl2br',           # New line to <br>
        'sane_lists'       # Better list handling
    ])

    # Convert markdown to HTML
    html_body = md.convert(md_content)

    # Get title
    title = metadata.get('title', md_file.stem)

    # Create complete HTML
    html = create_html_template(title, html_body, metadata)

    # Write HTML file
    output_file = output_dir / f"{md_file.stem}.html"
    output_file.write_text(html, encoding='utf-8')

    print(f"✅ Converted: {md_file.name} → {output_file.name}")


def main():
    """Convert all English markdown files in all four series plus top-level index."""

    # Base paths
    base_dir = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage")
    en_dir = base_dir / "wp" / "knowledge" / "en"

    # All four series
    series = ["mi-introduction", "nm-introduction", "pi-introduction", "mlp-introduction"]

    total_converted = 0

    print("🔄 Converting English Markdown files to HTML...")
    print(f"📁 Base directory: {en_dir}\n")

    # Convert top-level index first
    top_index = en_dir / "index.md"
    if top_index.exists():
        print("📚 Processing top-level index:")
        try:
            convert_markdown_to_html(top_index, en_dir)
            total_converted += 1
            print(f"✅ Converted: index.md → index.html")
        except Exception as e:
            print(f"❌ Error converting top-level index: {e}")

    # Convert series files
    for series_name in series:
        series_dir = en_dir / series_name

        if not series_dir.exists():
            print(f"⚠️  Directory not found: {series_dir}")
            continue

        # Find all markdown files
        md_files = sorted(series_dir.glob("*.md"))

        if not md_files:
            print(f"⚠️  No markdown files found in {series_name}")
            continue

        print(f"\n📚 Processing {series_name}:")
        print(f"   Found {len(md_files)} markdown file(s)")

        # Convert each file
        for md_file in md_files:
            try:
                convert_markdown_to_html(md_file, series_dir)
                total_converted += 1
            except Exception as e:
                print(f"❌ Error converting {md_file.name}: {e}")

    print(f"\n\n✅ Conversion complete! {total_converted} file(s) processed.")
    print(f"\n🌐 HTML files are now ready for GitHub Pages:")
    print(f"   https://yusukehashimotolab.github.io/wp/knowledge/en/index.html (Portal)")
    for series_name in series:
        print(f"   https://yusukehashimotolab.github.io/wp/knowledge/en/{series_name}/index.html")


if __name__ == "__main__":
    main()
