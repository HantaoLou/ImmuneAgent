#!/usr/bin/env python
"""
Download and store immunology papers locally in kb/data folder
This creates a local backup of the knowledge base
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from tqdm import tqdm

# Create data directory in kb
KB_DATA_PATH = Path(__file__).parent.parent.parent.parent / "kb" / "data"
PAPERS_PATH = KB_DATA_PATH / "papers"
METADATA_PATH = KB_DATA_PATH / "metadata"


def create_directories():
    """Create necessary directories for storing papers"""
    KB_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PAPERS_PATH.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.mkdir(parents=True, exist_ok=True)

    # Create category subdirectories
    categories = [
        "car_t_therapy",
        "checkpoint_inhibitors",
        "antibody_engineering",
        "t_cell_biology",
        "b_cell_immunology",
        "tumor_immunology",
        "autoimmunity",
        "infectious_disease",
        "immunometabolism",
        "single_cell",
        "cytokines",
        "vaccines",
        "transplantation",
        "innate_immunity",
        "immunodeficiency",
    ]

    for category in categories:
        (PAPERS_PATH / category).mkdir(exist_ok=True)

    print(f"✅ Created directory structure at: {KB_DATA_PATH}")
    return True


def download_pubmed_papers(query: str, category: str, max_papers: int = 50):
    """
    Download papers from PubMed and save locally

    Args:
        query: Search query
        category: Category folder to save in
        max_papers: Number of papers to download
    """
    papers_saved = 0
    category_path = PAPERS_PATH / category

    try:
        # PubMed API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Search
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_papers,
            "retmode": "json",
            "sort": "relevance",
        }

        print(f"  🔍 Searching PubMed: {query}")
        response = requests.get(f"{base_url}esearch.fcgi", params=search_params)
        search_results = response.json()

        id_list = search_results.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return 0

        # Fetch abstracts
        for i in range(0, len(id_list), 10):
            batch_ids = id_list[i : i + 10]

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "rettype": "abstract",
            }

            response = requests.get(f"{base_url}efetch.fcgi", params=fetch_params)

            # Save each paper
            for pmid in batch_ids:
                # Save abstract
                paper_file = category_path / f"pubmed_{pmid}.txt"
                with open(paper_file, "w", encoding="utf-8") as f:
                    f.write(f"PubMed ID: {pmid}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"Category: {category}\n\n")
                    f.write("Abstract:\n")
                    f.write(response.text[:5000])  # Save first 5000 chars

                # Save metadata
                metadata = {
                    "pmid": pmid,
                    "source": "pubmed",
                    "query": query,
                    "category": category,
                    "downloaded": datetime.now().isoformat(),
                    "file": str(paper_file.relative_to(KB_DATA_PATH)),
                }

                metadata_file = METADATA_PATH / f"pubmed_{pmid}.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                papers_saved += 1

            time.sleep(0.5)  # Rate limiting

    except Exception as e:
        print(f"  ❌ Error: {e}")

    return papers_saved


def download_arxiv_papers(query: str, category: str, max_papers: int = 20):
    """
    Download papers from arXiv

    Args:
        query: Search query
        category: Category folder
        max_papers: Number of papers
    """
    papers_saved = 0
    category_path = PAPERS_PATH / category

    try:
        import arxiv

        print(f"  🔍 Searching arXiv: {query}")

        search = arxiv.Search(
            query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance
        )

        for paper in search.results():
            # Save paper abstract
            paper_id = paper.entry_id.split("/")[-1]
            paper_file = category_path / f"arxiv_{paper_id}.txt"

            with open(paper_file, "w", encoding="utf-8") as f:
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join([a.name for a in paper.authors])}\n")
                f.write(f"Published: {paper.published}\n")
                f.write(f"arXiv ID: {paper.entry_id}\n")
                f.write(f"PDF: {paper.pdf_url}\n\n")
                f.write(f"Abstract:\n{paper.summary}\n")

            # Save metadata
            metadata = {
                "arxiv_id": paper_id,
                "source": "arxiv",
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "category": category,
                "pdf_url": paper.pdf_url,
                "downloaded": datetime.now().isoformat(),
                "file": str(paper_file.relative_to(KB_DATA_PATH)),
            }

            metadata_file = METADATA_PATH / f"arxiv_{paper_id}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            papers_saved += 1

    except Exception as e:
        print(f"  ❌ Error: {e}")

    return papers_saved


def create_knowledge_base_index():
    """Create an index of all downloaded papers"""

    index = {
        "created": datetime.now().isoformat(),
        "categories": {},
        "total_papers": 0,
        "sources": {"pubmed": 0, "arxiv": 0, "other": 0},
    }

    # Scan all papers
    for category_dir in PAPERS_PATH.iterdir():
        if category_dir.is_dir():
            papers = list(category_dir.glob("*.txt"))
            index["categories"][category_dir.name] = len(papers)
            index["total_papers"] += len(papers)

            # Count by source
            for paper in papers:
                if paper.name.startswith("pubmed_"):
                    index["sources"]["pubmed"] += 1
                elif paper.name.startswith("arxiv_"):
                    index["sources"]["arxiv"] += 1
                else:
                    index["sources"]["other"] += 1

    # Save index
    index_file = KB_DATA_PATH / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n📊 Index created: {index_file}")
    print(f"   Total papers: {index['total_papers']}")
    print(f"   Categories: {len(index['categories'])}")

    return index


def download_immunology_knowledge_base():
    """Download a comprehensive immunology knowledge base"""

    print("=" * 60)
    print("📚 Downloading Immunology Knowledge Base Locally")
    print("=" * 60)

    # Create directories
    if not create_directories():
        return

    # Define download tasks
    download_tasks = [
        # CAR-T therapy
        ("CAR-T cell therapy cancer", "car_t_therapy", 30),
        ("chimeric antigen receptor", "car_t_therapy", 20),
        # Checkpoint inhibitors
        ("PD-1 PD-L1 checkpoint inhibitor", "checkpoint_inhibitors", 30),
        ("CTLA-4 immunotherapy", "checkpoint_inhibitors", 20),
        # Antibody engineering
        ("antibody engineering humanization", "antibody_engineering", 30),
        ("bispecific antibody", "antibody_engineering", 20),
        # T cell biology
        ("T cell exhaustion", "t_cell_biology", 30),
        ("T cell differentiation", "t_cell_biology", 20),
        # B cell immunology
        ("B cell development", "b_cell_immunology", 25),
        ("BCR repertoire", "b_cell_immunology", 25),
        # Tumor immunology
        ("tumor microenvironment immune", "tumor_immunology", 30),
        ("cancer immunotherapy", "tumor_immunology", 20),
        # Single-cell
        ("single cell RNA sequencing immunology", "single_cell", 30),
        # Vaccines
        ("mRNA vaccine", "vaccines", 25),
        ("cancer vaccine neoantigen", "vaccines", 25),
    ]

    total_downloaded = 0

    # Download from PubMed
    print("\n📄 Downloading from PubMed...")
    for query, category, max_papers in download_tasks:
        count = download_pubmed_papers(query, category, max_papers)
        total_downloaded += count
        print(f"   ✅ {category}: {count} papers")
        time.sleep(1)  # Be nice to PubMed

    # Download from arXiv (computational papers)
    print("\n📄 Downloading from arXiv...")
    arxiv_queries = [
        ("immunology machine learning", "single_cell", 10),
        ("antibody design AI", "antibody_engineering", 10),
        ("T cell computational model", "t_cell_biology", 10),
    ]

    for query, category, max_papers in arxiv_queries:
        count = download_arxiv_papers(query, category, max_papers)
        total_downloaded += count
        print(f"   ✅ {category}: {count} papers")

    # Create index
    index = create_knowledge_base_index()

    # Create README
    readme_content = f"""# Immunology Knowledge Base

## Overview
This folder contains locally downloaded immunology papers for the ImmuneAgent system.

## Statistics
- **Total Papers**: {index["total_papers"]}
- **Categories**: {len(index["categories"])}
- **Sources**: PubMed ({index["sources"]["pubmed"]}), arXiv ({index["sources"]["arxiv"]})
- **Last Updated**: {index["created"]}

## Structure
```
kb/data/
├── papers/           # Paper abstracts organized by category
│   ├── car_t_therapy/
│   ├── checkpoint_inhibitors/
│   ├── antibody_engineering/
│   └── ...
├── metadata/         # JSON metadata for each paper
└── index.json        # Complete index of all papers
```

## Usage
These papers can be:
1. Loaded into Qdrant using `load_local_papers_to_qdrant.py`
2. Searched locally without Qdrant
3. Used as backup when Qdrant is unavailable

## Updating
Run `download_papers_locally.py` to update with latest papers.
"""

    readme_file = KB_DATA_PATH / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print(f"   Location: {KB_DATA_PATH}")
    print(f"   Total papers: {total_downloaded}")
    print(f"   Categories: {len(index['categories'])}")
    print("\n📝 Next steps:")
    print("   1. Run load_local_papers_to_qdrant.py to load into Qdrant")
    print("   2. Papers are now available locally as backup")
    print("=" * 60)


if __name__ == "__main__":
    download_immunology_knowledge_base()
