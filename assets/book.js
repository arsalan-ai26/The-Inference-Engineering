/* Shared runtime for every chapter: theme, contents, prev/next, quizzes, copy buttons, SVG helpers. */
(function () {
  const root = document.documentElement;
  const NS = 'http://www.w3.org/2000/svg';
  const themeFns = [];
  const IE = (window.IE = {
    css: (n) => getComputedStyle(root).getPropertyValue(n).trim(),
    el(tag, attrs, parent, text) {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
      if (text != null) e.textContent = text;
      if (parent) parent.appendChild(e);
      return e;
    },
    onTheme(fn) { themeFns.push(fn); fn(); },
    $: (id) => document.getElementById(id),
    int: (n) => Math.round(n).toLocaleString('en-US'),
    num(n, d) { return Number(n).toLocaleString('en-US', { maximumFractionDigits: d == null ? 2 : d, minimumFractionDigits: 0 }); },
    gb(b) { const g = b / 1e9; return g >= 100 ? g.toFixed(0) : g >= 10 ? g.toFixed(1) : g >= 1 ? g.toFixed(2) : g.toFixed(3); },
    tok(n) { return n >= 1024 && n % 1024 === 0 ? n / 1024 + 'K' : Math.round(n).toLocaleString('en-US'); },
  });
  const rerun = () => themeFns.forEach((f) => { try { f(); } catch (e) { console.error(e); } });

  /* theme toggle */
  const btn = document.createElement('button');
  btn.className = 'themebtn'; btn.textContent = 'Theme'; btn.setAttribute('aria-label', 'Toggle colour theme');
  btn.onclick = () => {
    const cur = root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('ie-theme', next); } catch (e) {}
    rerun();
  };
  try { const t = localStorage.getItem('ie-theme'); if (t) root.setAttribute('data-theme', t); } catch (e) {}
  document.body.appendChild(btn);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerun);

  /* contents from h2[id] */
  const nav = document.querySelector('nav.toc');
  if (nav) {
    const ch = document.body.dataset.chapter || '';
    const part = ch.split('.')[0];
    const ol = document.createElement('ol'); ol.id = 'tocList';
    document.querySelectorAll('main h2[id]').forEach((h) => {
      const li = document.createElement('li');
      const a = document.createElement('a'); a.href = '#' + h.id; a.textContent = h.textContent;
      li.appendChild(a); ol.appendChild(li);
    });
    nav.innerHTML = `<div class="book"><a href="../index.html" style="color:inherit">The Inference Engineering Book</a> · ${part ? 'Part ' + part : 'Guide'}</div>`;
    nav.appendChild(ol);
    const links = [...ol.querySelectorAll('a')];
    const obs = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
    }), { rootMargin: '-10% 0px -80% 0px' });
    document.querySelectorAll('main h2[id]').forEach((h) => obs.observe(h));
  }

  /* previous / next */
  const foot = document.querySelector('.navfoot');
  if (foot && window.IE_CHAPTERS) {
    const list = window.IE_CHAPTERS.filter((c) => c.ready);
    const i = list.findIndex((c) => c.n === document.body.dataset.chapter);
    const prev = list[i - 1], next = list[i + 1];
    foot.innerHTML = (prev ? `<a href="${prev.slug}.html">← ${prev.n} ${prev.title}</a>` : '<a href="../index.html">← Contents</a>') +
      (next ? `<a href="${next.slug}.html">${next.n} ${next.title} →</a>` : '<a href="../index.html">Contents →</a>');
  }

  /* quizzes: <div class="widget quiz" data-quiz="id"></div> + <script type="application/json" id="id"> */
  document.querySelectorAll('[data-quiz]').forEach((box) => {
    const Q = JSON.parse(document.getElementById(box.dataset.quiz).textContent);
    box.innerHTML = `<div class="wt">${Q.length} questions</div><div class="ws">Pick an answer to see why.</div>` +
      Q.map((x, i) => `<div class="q"><div><strong>${i + 1}.</strong> ${x.q}</div><div class="opts">${x.o.map((o, j) =>
        `<button data-j="${j}">${String.fromCharCode(97 + j)}) ${o}</button>`).join('')}</div><div class="fb" hidden></div></div>`).join('');
    box.querySelectorAll('.q').forEach((div, i) => {
      const x = Q[i], bs = div.querySelectorAll('button');
      bs.forEach((b) => (b.onclick = () => {
        bs.forEach((y) => y.classList.remove('right', 'wrong'));
        const j = +b.dataset.j;
        b.classList.add(j === x.a ? 'right' : 'wrong');
        if (j !== x.a) bs[x.a].classList.add('right');
        const fb = div.querySelector('.fb'); fb.hidden = false;
        fb.textContent = (j === x.a ? 'Correct. ' : 'Not quite. ') + x.e;
      }));
    });
  });

  /* copy buttons on code blocks */
  document.querySelectorAll('pre').forEach((pre) => {
    if (pre.classList.contains('out')) return;
    const b = document.createElement('button'); b.className = 'copybtn'; b.textContent = 'Copy';
    b.onclick = async () => {
      const text = [...pre.childNodes].filter((n) => n !== b).map((n) => n.textContent).join('');
      try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; }
      catch (e) { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = 'Selected'; }
      setTimeout(() => (b.textContent = 'Copy'), 1500);
    };
    pre.prepend(b);
  });

  window.addEventListener('load', rerun);
})();
