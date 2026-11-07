"""Build the site: wrap src/<slug>.html fragments into chapter pages, write assets/chapters.js and index.html.
Run: python3 tools/build.py"""
import json, os, re, html
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(ROOT, 'chapters.json')))
FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap">'
def head(title, prefix):
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
{FONTS}
<link rel="stylesheet" href="{prefix}assets/book.css">
</head>'''
chapters = []
for part in data['parts']:
    for n, title, slug in part['chapters']:
        ready = bool(slug) and os.path.exists(os.path.join(ROOT, 'src', slug + '.html'))
        chapters.append({'n': n, 'title': title, 'slug': slug, 'part': part['n'], 'ready': ready})
open(os.path.join(ROOT, 'assets', 'chapters.js'), 'w').write('window.IE_CHAPTERS = ' + json.dumps(chapters, indent=0) + ';\n')
for c in chapters:
    if not c['ready']: continue
    frag = open(os.path.join(ROOT, 'src', c['slug'] + '.html')).read()
    scripts = ''
    m = re.search(r'<!--SCRIPT-->(.*)$', frag, re.S)
    if m: scripts, frag = m.group(1), frag[:m.start()]
    page = head(f"{c['n']} {c['title']}", '../') + f'''
<body data-chapter="{c['n']}">
<div class="topbar" style="padding-left:0;padding-right:0"><a href="../index.html">← The Inference Engineering Book</a><span>Part {c['part']} · Chapter {c['n']}</span></div>
<div class="shell">
<nav class="toc" aria-label="Chapter contents"></nav>
<main><article>
{frag}
<div class="navfoot"></div>
</article></main>
</div>
<script src="../assets/chapters.js"></script>
<script src="../assets/book.js"></script>
{scripts}
</body>
</html>
'''
    open(os.path.join(ROOT, 'chapters', c['slug'] + '.html'), 'w').write(page)
# guides: src/guides/<name>.html -> guides/<name>.html
gdir = os.path.join(ROOT, 'src', 'guides')
if os.path.isdir(gdir):
    os.makedirs(os.path.join(ROOT, 'guides'), exist_ok=True)
    for fn in sorted(os.listdir(gdir)):
        if not fn.endswith('.html'): continue
        frag = open(os.path.join(gdir, fn)).read()
        scripts = ''
        m = re.search(r'<!--SCRIPT-->(.*)$', frag, re.S)
        if m: scripts, frag = m.group(1), frag[:m.start()]
        t = re.search(r'<h1>(.*?)</h1>', frag, re.S).group(1)
        page = head(re.sub('<[^>]+>', '', t), '../') + f'''
<body data-chapter="">
<div class="topbar" style="padding-left:0;padding-right:0"><a href="../index.html">← The Inference Engineering Book</a><span>Guide</span></div>
<div class="shell">
<nav class="toc" aria-label="Contents"></nav>
<main><article>
{frag}
</article></main>
</div>
<script src="../assets/chapters.js"></script>
<script src="../assets/book.js"></script>
{scripts}
</body>
</html>
'''
        open(os.path.join(ROOT, 'guides', fn), 'w').write(page)
# index
cards = []
for part in data['parts']:
    lis = []
    for n, title, slug in part['chapters']:
        c = next(x for x in chapters if x['n'] == n)
        if c['ready']:
            lis.append(f'<li class="ready"><span class="num">{n}</span><a href="chapters/{slug}.html">{html.escape(title)}</a><span class="tag">ready</span></li>')
        else:
            lis.append(f'<li><span class="num">{n}</span><span>{html.escape(title)}</span></li>')
    cards.append(f'<section class="part" id="part-{part["n"]}"><div class="pn">Part {part["n"]} · {part["weeks"]}</div><h2>{html.escape(part["title"])}</h2><ol>{"".join(lis)}</ol></section>')
extra = ''.join(f'<li><a href="{href}">{t}</a></li>' for href, t in [('guides/start-here.html','Chapter 0: from agent builder to inference engineer (start here)'),('guides/compute.html','Compute guide: free GPUs, rentals and costs'),('guides/glossary.html','Glossary'),('guides/formulas.html','Formula sheet'),('guides/reading-list.html','Reading list: every paper the book cites')] if os.path.exists(os.path.join(ROOT, href)))
idx = head('The Inference Engineering Book', '') + f'''
<body class="index">
<div class="iwrap">
<h1>The Inference Engineering Book</h1>
<p class="lede">From AI engineer to inference expert, in eight parts. Each chapter starts from first principles and builds step by step, with interactive visuals, hands-on labs that run on a CPU or a free Colab GPU (each result labelled as measured, and GPU-only parts labelled as expected), worked numbers on real models, and references at the end. New here and coming from agent and LLM application work? Start with <a href="guides/start-here.html">Chapter 0: from agent builder to inference engineer</a>.</p>
<div class="legend"><span><b class="t-pre">Prefill</b> is indigo</span><span><b class="t-dec">Decode</b> is amber</span><span><b class="t-kv">KV cache</b> is teal</span><span>The same colours are used in every chapter.</span></div>
{('<div class="box real"><span class="lbl">Guides</span><ul>' + extra + '</ul></div>') if extra else ''}
<div class="igrid">{''.join(cards)}</div>
<p class="note" style="margin-top:40px">All 62 chapters are complete (four in Part 1 are optional math for reading papers). Facts, defaults and prices carry the date they were checked.</p>
</div>
<script src="assets/book.js"></script>
</body>
</html>
'''
open(os.path.join(ROOT, 'index.html'), 'w').write(idx)
print('ready:', [c['n'] for c in chapters if c['ready']])
