import argparse
import asyncio
import os
from doc_crawler import DocCrawler

def main():
    parser = argparse.ArgumentParser(description='Crawl and process documentation using AI')
    parser.add_argument('url', help='The base URL to start crawling from')
    parser.add_argument('--output', default='docs', help='Output directory for processed documentation')
    parser.add_argument('--concurrent', type=int, default=3, help='Number of concurrent pages to scrape')
    parser.add_argument('--model', choices=['gpt', 'gemini'], default='gpt', help='AI model to use for processing')
    parser.add_argument('--browser', choices=['chrome', 'firefox'], default='chrome', help='Browser to use for scraping')
    parser.add_argument('--custom-prompt', help='Path to a text file containing custom prompt for the AI model')
    parser.add_argument('--max-pages', type=int, default=1000, help='Maximum number of pages to crawl')
    
    args = parser.parse_args()
    
    # Read custom prompt from file if specified
    custom_prompt = None
    if args.custom_prompt:
        try:
            with open(args.custom_prompt, 'r', encoding='utf-8') as f:
                custom_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading custom prompt file: {str(e)}")
            return
    
    crawler = DocCrawler(
        base_url=args.url,
        output_dir=args.output,
        concurrent_pages=args.concurrent,
        model_type=args.model,
        browser_type=args.browser,
        custom_prompt=custom_prompt,
        max_pages=args.max_pages
    )
    
    asyncio.run(crawler.crawl())

if __name__ == '__main__':
    main() 