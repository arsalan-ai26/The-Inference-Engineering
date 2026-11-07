"""Build src/guides/reading-list.html from the reference tables of every chapter.
Each arXiv paper cited in a chapter's References table is listed once, under the part of the
first chapter that cites it, with that chapter's one-line description. LAB and CORE below add the lab tag and tier."""
import json, os, re, html
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = {  # main affiliation(s); "see paper" where not confirmed
'2407.01082':'Academic','2209.05433':'NVIDIA, Arm, Intel','1910.07467':'Academic','2002.05202':'Google','1512.03385':'Microsoft',
'1508.07909':'Academic','1706.03762':'Google','2205.14135':'Stanford','2309.17453':'MIT, Meta','2104.09864':'Zhuiyi Technology',
'2306.15595':'Meta','2309.00071':'Nous Research and others','2309.06180':'Berkeley','2305.13245':'Google','1911.02150':'Google',
'2405.04434':'DeepSeek','2609.19969':'DeepSeek','2412.19437':'DeepSeek','2401.04088':'Mistral AI','2509.01322':'Meituan',
'2507.20534':'Moonshot','2312.00752':'CMU, Princeton','2510.26692':'Moonshot','2412.06464':'NVIDIA, MIT','2501.08313':'MiniMax',
'2506.13585':'MiniMax','2606.19348':'DeepSeek','2505.09388':'Alibaba','2508.06471':'Zhipu','2502.09992':'Renmin University, Ant Group',
'2503.09573':'Academic','2510.01123':'see paper','2509.20241':'see paper','2512.24880':'DeepSeek','2512.14693':'see paper',
'2403.02310':'Microsoft Research and others','2401.00588':'Berkeley, Stanford','2406.03243':'Alibaba','2503.18292':'Academic (vLLM project)',
'2405.04437':'Microsoft Research','2408.12757':'University of Washington','2312.07104':'Stanford, Berkeley (LMSYS)','2411.15100':'CMU, NVIDIA',
'2508.15601':'Shanghai AI Laboratory','2402.02750':'Academic','2405.04532':'MIT, NVIDIA','2404.00456':'ETH Zurich and others','2306.07629':'Berkeley',
'2210.17323':'IST Austria','2306.00978':'MIT','2211.10438':'MIT, NVIDIA','2502.11089':'DeepSeek, Peking University','2502.13189':'Moonshot',
'2307.08691':'Princeton, Stanford','2407.08608':'Princeton, NVIDIA, Meta and others','2603.05451':'see paper','2501.01005':'University of Washington, NVIDIA',
'2512.02556':'DeepSeek','2305.09781':'CMU','2402.02057':'UCSD, Berkeley','2609.24698':'see paper','2211.17192':'Google','2401.10774':'Princeton and others',
'2503.01840':'Peking University and others','2411.04975':'Snowflake, CMU','2402.05099':'Stanford','2310.07240':'University of Chicago','2510.09665':'University of Chicago',
'2407.00079':'Moonshot, Tsinghua University','2405.16444':'University of Chicago','2502.19811':'ByteDance','2504.02263':'ByteDance','2507.19427':'StepFun',
'2506.12708':'Huawei','2508.02520':'Huawei','2507.07120':'NVIDIA','1909.08053':'NVIDIA','2401.09670':'Peking University, UCSD','2311.18677':'Microsoft',
'2404.06654':'NVIDIA','2310.01889':'Berkeley','2407.02490':'Microsoft','2501.15383':'Alibaba','2306.14048':'Academic','2404.14469':'Academic',
'2505.09343':'DeepSeek','2408.14158':'DeepSeek','2405.05254':'Microsoft, Tsinghua University','2606.09079':'see paper','2608.01662':'Meituan',
'2504.17577':'Peking University, Microsoft Research','2407.21783':'Meta','2409.03992':'see paper','2410.21680':'Meta','2502.07776':'Stanford',
'2501.12948':'DeepSeek','2312.01552':'Allen AI, University of Washington','2106.09685':'Microsoft','2305.14314':'University of Washington',
'2205.05198':'NVIDIA','2402.03300':'DeepSeek','2305.18290':'Stanford','2509.14234':'see paper','2311.03285':'Berkeley, Stanford','1910.02054':'Microsoft'}
CN = {'DeepSeek','Moonshot','Alibaba','MiniMax','StepFun','Meituan','ByteDance','Huawei','Zhipu','Shanghai AI Laboratory','Zhuiyi Technology','Peking University','Tsinghua University','Renmin University','Ant Group'}
CORE = set('''1706.03762 2205.14135 2104.09864 2309.06180 2305.13245 2405.04434 2412.19437 2401.04088 2312.00752 2501.08313 2606.19348 2609.19969
2403.02310 2312.07104 2411.15100 2210.17323 2306.00978 2211.10438 2307.08691 2407.08608 2502.11089 2512.02556 2211.17192 2503.01840 2305.09781
2407.00079 2402.05099 1909.08053 2401.09670 2504.02263 2507.19427 2506.12708 2505.09343 2504.17577 2501.12948 2106.09685 2305.18290 2402.03300'''.split())
EXTRA = [  # non-arXiv papers cited in the book: (part, chapter, title, url, why, lab, core)
(3,'3.2','Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)','https://www.usenix.org/conference/osdi22/presentation/yu','Iteration-level scheduling and selective batching: the origin of continuous batching.','Seoul National University, FriendliAI',True),
(3,'3.6','Chen et al., KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models (SOSP 2025)','https://github.com/kvcache-ai/ktransformers','CPU and GPU hybrid inference of large MoE models.','Tsinghua University, Approaching.AI',False),
(6,'6.3','Xiang et al., Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market (SOSP 2025)','https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf','Token-level autoscaling across many models; 82% fewer GPUs in production.','Peking University, Alibaba',True),
]
c = json.load(open(os.path.join(ROOT, 'chapters.json')))
PART = {p['n']: p['title'] for p in c['parts']}
rows = {}; order = []
for p in c['parts']:
    for num, title, slug in p['chapters']:
        s = open(os.path.join(ROOT, 'src', slug + '.html')).read()
        i = s.find('id="refs"'); r = s[i:] if i > 0 else ''
        for m in re.finditer(r'<tr><td>(.*?)</td><td>(.*?)</td></tr>', r, re.S):
            for a in re.finditer(r'<a href="(https://arxiv\.org/abs/([0-9.]+))">(.*?)</a>', m.group(1)):
                aid = a.group(2)
                if aid in rows: continue
                rows[aid] = (p['n'], num, slug, re.sub('<[^>]+>', '', a.group(3)), a.group(1), re.sub('<[^>]+>', '', m.group(2)).strip())
                order.append(aid)
missing = [a for a in order if a not in LAB]
assert not missing, f'add lab tags for {missing}'
def is_cn(lab): return any(x in lab for x in CN)
items = {n: [] for n in PART}
for aid in order:
    pn, num, slug, t, u, why = rows[aid]
    items[pn].append((num, slug, t, u, why, LAB[aid], aid in CORE))
for pn, num, t, u, why, lab, core in EXTRA:
    slug = next(s for n, _, s in sum((p['chapters'] for p in c['parts']), []) if n == num)
    items[pn].append((num, slug, t, u, why, lab, core))
key = lambda x: [int(v) for v in x[0].split('.')]
total = sum(len(v) for v in items.values()); ncn = sum(is_cn(x[5]) for v in items.values() for x in v)
ncore = sum(x[6] for v in items.values() for x in v)
out = ['<div class="kicker">Guide</div>', '<h1>Reading list</h1>',
 f'<p class="lede">Every research paper the book cites, in one place: {total} papers, grouped by the part that first uses them and ordered as a reader should meet them. Each has one line on why to read it (from the chapter that cites it), the lab it comes from, and a tier. <strong>Core</strong> ({ncore} papers) are the ones a practitioner should have read; <strong>extended</strong> are for depth on one topic. {ncn} come from Chinese labs and universities, whose inference work is now some of the most detailed published.</p>',
 '<div class="box def"><span class="lbl">How to read the Chinese-lab technical reports</span><p>Model reports from DeepSeek, Moonshot, Alibaba, MiniMax and others are long (often 50 pages or more) and mostly about training data and benchmarks. The inference content is usually in one section: <strong>DeepSeek-V3</strong>, section 3.4 "Inference and Deployment"; <strong>DeepSeek-V4</strong>, section 3.5 "Inference Framework" (KV layout, on-disk cache) and 3.1 and 3.3 (expert overlap, batch-invariant kernels); <strong>DeepSeek-V4.1-Flash</strong>, section 3.2 "Inference System" (persistent KV, bounded replay) and 2.4 (DSpark, FP4 KV); <strong>MiniMax-01</strong>, section 3.3 "Lightning Attention Inference Optimization"; <strong>Insights into DeepSeek-V3</strong>, section 2.3 "Increasing Inference Speed". Systems papers (Step-3, MegaScale-Infer, CloudMatrix384) are about inference throughout. Read the abstract, the architecture section and the inference section first, then the evaluation tables for the numbers you need.</p></div>',
 '<p class="note">Lab tags give the main affiliation listed on each paper, to help you find a group\'s other work; "see paper" marks ones not confirmed here. The list is generated from the chapters\' reference tables by <code>tools/gen_reading_list.py</code>.</p>']
for pn in sorted(items):
    v = sorted(items[pn], key=key)
    if not v: continue
    out.append(f'<h2 id="part{pn}">Part {pn}. {html.escape(PART[pn])}</h2>')
    out.append('<div class="tbl"><table><tr><th>Paper</th><th>Why read it</th><th>Lab</th><th>Tier</th><th>Chapter</th></tr>')
    for num, slug, t, u, why, lab, core in v:
        tag = lab + (' <span class="chip">CN</span>' if is_cn(lab) else '')
        out.append(f'<tr><td><a href="{u}">{t}</a></td><td>{why}</td><td>{tag}</td><td>{"core" if core else "extended"}</td><td><a href="../chapters/{slug}.html">{num}</a></td></tr>')
    out.append('</table></div>')
out.append('<p class="note">Generated on 7 November 2026 from the reference tables of all chapters.</p>')
open(os.path.join(ROOT, 'src', 'guides', 'reading-list.html'), 'w').write('\n'.join(out) + '\n')
print(f'{total} papers, {ncore} core, {ncn} from Chinese labs')
