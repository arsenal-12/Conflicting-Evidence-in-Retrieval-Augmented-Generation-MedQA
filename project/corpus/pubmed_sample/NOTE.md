# Data Provenance Note

The `.json` files in this directory are **synthetically generated
placeholder abstracts**, formatted to exactly match the `Document` schema
this pipeline expects (`document_id`, `title`, `text`, `source`, `year`,
`metadata`). They were written to plausible, medically-themed content so
the retrieval pipeline has something meaningful to index and search
end-to-end out of the box, since downloading real PubMed abstracts
requires network access (e.g. via the NCBI E-utilities API) and is subject
to NCBI's usage policies.

**Before using this repository for real dissertation experiments**,
replace these files with genuine PubMed abstracts:

1. Retrieve abstracts via the NCBI E-utilities API
   (<https://www.ncbi.nlm.nih.gov/books/NBK25501/>) or a PubMed bulk
   dataset.
2. Convert each abstract into a JSON file matching the `Document` schema
   above (one file per document, as in this directory).
3. Replace the contents of `corpus/pubmed_sample/` with the converted
   files.

No code changes are required — `DocumentLoader` only depends on files
matching this schema, not on how they were produced.
