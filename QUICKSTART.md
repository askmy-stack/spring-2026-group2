# LLM Wiki: Quick Start

Get your research wiki running in 5 minutes.

## Step 1: Add Your First Paper

Find a paper on EEG seizure detection (arxiv-sanity-lite, arXiv, or local PDF).

Create a file in `sources/`:

```bash
mkdir -p sources/lstm-variants
# Then add your paper (as text, markdown, or PDF reference)
```

Example: `sources/lstm-variants/2024-attention-lstm-seizure.md`

```markdown
# Attention-Enhanced LSTM for Seizure Detection

**Authors:** Smith et al.
**Year:** 2024
**Link:** https://arxiv.org/abs/2024.xxxxx

## Summary

This paper proposes an attention mechanism on top of LSTM for EEG seizure detection...
[Your notes as you read]
```

## Step 2: Run the Ingest Workflow

Tell me:
> "Ingest the paper at `sources/lstm-variants/2024-attention-lstm-seizure.md`"

I will:
1. Read it thoroughly
2. Extract key claims and findings
3. Create a source summary page: `wiki/sources/2024-attention-lstm-seizure.md`
4. Create/update entity pages (Models, Techniques, Datasets mentioned)
5. Update `wiki/index.md` with new pages
6. Append to `wiki/log.md`

## Step 3: Review & Refine

You'll see:
- Summary of the paper
- New pages created (or updated)
- Links between entities

You can:
- Ask me to emphasize different aspects
- Request comparisons with other papers
- Identify contradictions or gaps
- Commit the changes

## Example Ingest Session

You: "Ingest `sources/lstm-variants/2024-attention-lstm-seizure.md`"

Me:
```
✓ Read paper
✓ Created wiki/sources/2024-attention-lstm-seizure.md
✓ Created wiki/entities/models/attention-lstm.md
✓ Updated wiki/entities/techniques/attention-mechanisms.md
✓ Updated wiki/index.md
✓ Appended log entry
```

You review the created pages, ask clarifying questions, approve.

## Step 4: Keep Going

Repeat Step 1-3 with more papers. The wiki compounds:
- Entities get richer (more cross-references, more data points)
- Topics emerge and clarify
- Patterns (what works, what doesn't) become visible
- Contradictions surface and get flagged

## Available Commands

Once you have papers in `sources/`:

- **"Ingest `<path>`"** — Run full ingest workflow
- **"Query: [question]"** — Ask the wiki a question (searches pages, synthesizes, cites)
- **"Lint the wiki"** — Check for contradictions, orphan pages, gaps
- **"Show me the architecture comparison"** — Run a specific synthesis

## Tips

1. **Start small** — 2–3 papers first. See how the wiki grows before adding 50.
2. **Use arxiv-sanity-lite** — To find papers, browse topics, get recommendations. Then feed them here.
3. **Organize sources by topic** — Easier to navigate `sources/lstm-variants/` than a flat list.
4. **Review wiki/ often** — Especially early. Check the graph, follow links, see what's connected.
5. **Ask me questions** — "Why does model X outperform Y?" triggers wiki synthesis and new pages.

---

**Ready to start?** Add your first paper and say "ingest."
