"""Generate src/guides/glossary.html from every chapter's definition boxes plus tools/gloss_extra.json.

A definition is any <strong>Term:</strong> text inside a <div class="box def"> block. The first chapter
(in book order) that defines a term wins. Run before tools/build.py.
"""
import html, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
book = json.load(open(os.path.join(ROOT, "chapters.json")))


def chapters():
    def walk(x):
        if isinstance(x, list):
            if len(x) == 3 and all(isinstance(v, str) for v in x) and re.match(r"^\d+\.\d+$", x[0]):
                yield x
            else:
                for v in x:
                    yield from walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
    yield from walk(book)


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s)


entries = {}
SKIP = {"the rule", "memory per sequence is constant", "decode cost per token is constant"}
for num, title, slug in chapters():
    path = os.path.join(ROOT, "src", slug + ".html")
    if not os.path.exists(path):
        continue
    src = open(path, encoding="utf-8").read()
    for box in re.findall(r'<div class="box def">(.*?)</div>', src, flags=re.S):
        for para in re.findall(r"<p>(.*?)</p>", box, flags=re.S):
            # one paragraph may hold several "<strong>Term:</strong> text" pairs
            # forms: "<strong>Term:</strong> text" and "<strong>Term</strong> (qualifier): text"
            marks = list(re.finditer(r"<strong>([^<]{1,90}?)(?::</strong>|</strong>(\s*\([^)]{0,80}\))\s*:)", para))
            for i, m in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(para)
                term = strip_tags(m.group(1)).strip()
                text = ((m.group(2).strip() + ": ") if m.group(2) else "") + para[m.end():end].strip()
                if term.lower() in SKIP:
                    continue
                if not term or not text:
                    continue
                key = term.lower()
                if key not in entries:
                    entries[key] = (term, text, num, slug)

extra_path = os.path.join(ROOT, "tools", "gloss_extra.json")
slug_of = {n: s for n, t, s in chapters()}
for term, text, num in json.load(open(extra_path, encoding="utf-8")):
    key = term.lower()
    if key not in entries and num in slug_of:
        entries[key] = (term, text, num, slug_of[num])

by_letter = {}
for key in sorted(entries, key=lambda k: re.sub(r"[^a-z0-9]", "", k)):
    term, text, num, slug = entries[key]
    first = re.sub(r"[^A-Za-z]", "", term)[:1].upper() or "#"
    by_letter.setdefault(first, []).append((term, text, num, slug))

out = ['<style>dl.gl{margin:0}.ge{padding:8px 0;border-bottom:1px solid var(--line, #ddd)}.ge dt{font-weight:600}'
       '.ge dd{margin:2px 0 0}.chref{font-size:.85em;white-space:nowrap}</style>',
       '<div class="kicker">Guide</div>', '<h1>Glossary</h1>',
       '<p class="lede">Every term this book defines, in one place. Each entry links to the chapter that '
       'explains it with examples. Use the search box to filter.</p>',
       '<div class="widget"><div class="ctrls"><label>Filter <input type="search" id="gq" placeholder="for example: cache" '
       'style="width:220px"></label></div><div class="ws" id="gcount"></div></div>']
for letter in sorted(by_letter):
    out.append(f'<h2 id="l-{letter.lower()}">{letter}</h2><dl class="gl">')
    for term, text, num, slug in by_letter[letter]:
        out.append(f'<div class="ge"><dt>{term}</dt><dd>{text} <a class="chref" href="../chapters/{slug}.html">'
                   f'Chapter {num}</a></dd></div>')
    out.append("</dl>")
out.append('<h2 id="refs">References</h2><p>Definitions are condensed from the chapters of this book; each chapter\'s '
           'own references list the sources behind them.</p>')
out.append("<!--SCRIPT-->\n<script>\n(function(){const q=IE.$('gq'),es=[...document.querySelectorAll('.ge')];function f(){"
           "const v=q.value.trim().toLowerCase();let c=0;es.forEach(e=>{const ok=!v||e.textContent.toLowerCase().includes(v);"
           "e.hidden=!ok;if(ok)c++;});document.querySelectorAll('dl.gl').forEach(dl=>{const h=dl.previousElementSibling;"
           "const any=[...dl.children].some(x=>!x.hidden);dl.hidden=!any;h.hidden=!any;});IE.$('gcount').textContent=c+' of '"
           "+es.length+' terms';}q.addEventListener('input',f);f();})();\n</script>")
open(os.path.join(ROOT, "src", "guides", "glossary.html"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(len(entries), "terms")
