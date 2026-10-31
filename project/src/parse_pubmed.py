import gzip
import json
from pathlib import Path
from lxml import etree
from tqdm import tqdm

RAW_FOLDER = Path("corpus/pubmed_sample/pubmed_raw")
OUTPUT_FILE = Path("corpus/pubmed_50k.jsonl")

files = sorted(RAW_FOLDER.glob("*.xml.gz"))

count = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as out:

    for file in files:
        print(f"Processing {file.name}")

        with gzip.open(file, "rb") as f:

            context = etree.iterparse(
                f,
                events=("end",),
                tag="PubmedArticle"
            )

            for _, article in tqdm(context):

                try:

                    pmid = article.findtext(".//PMID")

                    title = article.findtext(".//ArticleTitle")

                    abstract_nodes = article.findall(".//AbstractText")

                    abstract = " ".join(
                        node.text.strip()
                        for node in abstract_nodes
                        if node.text
                    )

                    if not title:
                        article.clear()
                        continue

                    if not abstract:
                        article.clear()
                        continue

                    record = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract
                    }

                    out.write(json.dumps(record) + "\n")

                    count += 1

                except Exception:
                    pass

                article.clear()

print()
print(f"Finished.")
print(f"Extracted {count} abstracts.")
print(f"Saved to {OUTPUT_FILE}")