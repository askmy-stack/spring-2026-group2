# LLM Wiki Schema

This document defines the structure, conventions, and workflows for maintaining the knowledge base.

## Overview

The wiki is a persistent, compounding artifact that sits between raw sources and insights. It accumulates knowledge incrementally:
- **Sources** (immutable) — raw documents, articles, papers, notes
- **Wiki** (generated) — markdown pages that synthesize and cross-reference sources
- **Queries** — explorations that compound into the wiki

The LLM owns the wiki layer. You curate sources and ask questions; the LLM maintains structure, updates, and linkage.

---

## Architecture

### Directory Structure

```
project-root/
├── wiki/                    # Generated markdown pages
│   ├── index.md            # Content catalog (updated on each ingest)
│   ├── log.md              # Chronological log of operations
│   ├── entities/           # Domain concepts, systems, models
│   ├── topics/             # Research areas and themes
│   ├── sources/            # Source summaries
│   └── syntheses/          # Cross-cutting analyses
├── sources/                 # Raw immutable documents
│   └── (organized by user) 
└── WIKI.md                 # This schema file
```

---

## Page Types

Define the types of pages your wiki should contain. This shapes what the LLM creates and maintains.

### Domain-Specific: EEG Seizure Detection Research

#### Entities

- **Models** — Specific architectures (LSTM, CNN-LSTM, attention variants, etc.)
  - Fields: Name, year, authors, paper link, key innovation, performance metrics
- **Datasets** — EEG benchmark datasets (CHB-MIT, Temple University, etc.)
  - Fields: Name, seizure count, patient count, sampling rate, availability, paper link
- **Metrics** — Performance measures (sensitivity, specificity, AUC, false positive rate, etc.)
  - Fields: Name, definition, why it matters, typical ranges for SOTA
- **Techniques** — Technical approaches (attention mechanisms, class imbalance handling, feature extraction, etc.)
  - Fields: Name, description, when to use, papers using it

#### Topics

- LSTM architectures for time-series
- Convolutional approaches for EEG
- Attention mechanisms in seizure detection
- Class imbalance in medical datasets
- Feature extraction from EEG
- Transfer learning for EEG
- Real-time/edge seizure detection
- Interpretability in seizure detection models

#### Syntheses

- **Architecture Comparison** — Table: models, components, performance, year
- **Dataset Benchmark** — Which datasets, metrics reported, SOTA results per dataset
- **Open Problems** — Gaps in literature, unsolved challenges
- **Reproducibility Matrix** — Which papers publish code, which are reproducible

---

## Workflows

### 1. Ingest a Source

When you add a new document to `sources/`:

1. **Read and discuss** — LLM reads source, summarizes key takeaways
2. **Write source page** — Create `wiki/sources/<title>.md` with:
   - Summary of content
   - Key claims or findings
   - Links to related entity/topic pages
3. **Update entities/topics** — Revise affected pages with:
   - New information, contradictions, confirmations
   - Links to this source
4. **Update index** — Add new pages, mark updated pages with date
5. **Append log** — Record ingest: `## [YYYY-MM-DD] ingest | <Title>`

**Supervision:** Read summaries, approve updates, guide emphasis. (Can batch later with less oversight.)

### 2. Query the Wiki

When you ask a question:

1. **Search index** — LLM reads `wiki/index.md` to locate relevant pages
2. **Read pages** — Drill into full page content
3. **Synthesize** — Answer with citations to pages
4. **File answer** — If valuable, write the answer as a new wiki page
   - Comparisons, analyses, connection maps
   - These become new content, not chat ephemera

### 3. Lint the Wiki

Periodically (e.g., after 5–10 sources):

1. **Check for contradictions** — Conflicting claims across pages
2. **Mark stale claims** — Newer sources superseding old findings
3. **Find orphans** — Pages with no inbound links
4. **Identify gaps** — Mentioned concepts lacking their own page
5. **Suggest questions** — What should we investigate next?

---

## Conventions

### Markdown Style

- **Links:** Use `[text](wiki/path/to/page.md)` for internal links
- **Dates:** ISO 8601 format (`YYYY-MM-DD`)
- **Sections:** Use `##` (page title is `#`)
- **Metadata:** Optional YAML frontmatter if tracking author, date, version
- **Citations:** Link to source pages, e.g., `[source](wiki/sources/title.md)`

### File Naming

- Lowercase with hyphens: `entity-name.md`, `topic-area.md`
- Source pages: `wiki/sources/<title>.md` matching source file name if possible
- No spaces; prefer dashes for readability

### Updating Pages

When adding to existing pages:
- Append new information with clear attribution (e.g., "From [source](link)")
- Flag contradictions explicitly: "**Contradiction:** Previous claim was X, but [source] states Y"
- Date updates: "Updated 2026-04-11: [change]"
- Preserve old information; don't delete unless certain it's wrong

---

## Index and Log

### index.md

Content-oriented catalog. Updated on every ingest.

Format:
```
### Category Name

- [Page Title](wiki/path/to/page.md) — One-line summary, metadata if useful
```

Purpose: Lets LLM (and you) find relevant pages quickly; avoids embedding-based RAG for moderate scales.

### log.md

Append-only chronological record.

Format:
```
## [YYYY-MM-DD] event-type | Event Title

Brief description of what happened.
```

Event types: `ingest`, `query`, `lint`, `update`, `synthesis`

Parseable with grep: `grep "^## \[" log.md | tail -5` gives last 5 events.

---

## Starting Out

1. **Define page types** (this section, "Domain-Specific")
2. **Ingest first source** — Start with 1 article/document; go through workflow #1
3. **Ask a question** — Use workflow #2 to query and file the answer
4. **Iterate** — Add sources and queries until you have 5–10 pages
5. **Lint** — Run workflow #3 to clean up and identify gaps

---

## Notes

- The wiki is **persistent**. Every source you add, every answer you file compounds. The knowledge base grows richer and more connected with time.
- The LLM is **responsible for maintenance**. You guide; the LLM writes, cross-references, flags contradictions, keeps everything consistent.
- **No RAG infrastructure needed** at moderate scale. The index + full-text reading works well up to ~100 sources, ~hundreds of pages.
- **Obsidian integration** (optional): Keep `wiki/` synced to an Obsidian vault for visual browsing, link graphs, backlinks. The LLM generates markdown; you browse and explore in Obsidian.

---

## Configuration: EEG Seizure Detection

✅ **Page types defined** — See "Domain-Specific" section above

**Source organization:** 
- By topic area (e.g., `sources/attention-mechanisms/`, `sources/lstm-variants/`)
- Filename: `sources/<topic>/<year>-<title>.md`

**Update frequency:**
- Ingest as you find papers (interactive workflow — you read summary, approve updates)
- Batch lint every 5–10 papers

**Workflows:**
1. Find paper (arxiv-sanity-lite or direct)
2. Read abstract/introduction
3. Add to `sources/`
4. Run ingest workflow (LLM: summarize, link to entities, update index)
5. Review updates in wiki
6. Commit
