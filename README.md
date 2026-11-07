# The Inference Engineering Book

An interactive, chapter-by-chapter book that takes an AI engineer with little systems experience to expert-level LLM inference engineering.

Read it on GitHub Pages: open `index.html` from the published site.

- Eight parts: foundations, transformer inference mechanics, engines and serving, model optimization, GPUs and kernels, production infrastructure, fine-tuning, and specialized workloads.
- Every chapter has interactive visuals, a lab that runs on a CPU or a free Colab GPU (results labelled as measured, GPU-only parts as expected), source corrections, a quiz and checked references.
- Facts, defaults and prices carry the date they were checked.

## Structure

```
index.html            table of contents
chapters/<n>-<m>-<slug>.html   one self-contained page per chapter
guides/               Chapter 0 (start here), compute guide, glossary, formula sheet and reading list
labs/                 runnable lab code for Part 8 (mini inference engine, tests, benchmarks)
src/, tools/          chapter sources and the build and glossary scripts
```

No build step. Each page is plain HTML, CSS and JavaScript.
