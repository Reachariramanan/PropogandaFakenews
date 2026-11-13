const LABELS=['FAKE','PROPAGANDA','LEGIT','MISLEADING','INCONCLUSIVE'];
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const fmt=(x,d=3)=>x==null||isNaN(x)?'–':(+x).toFixed(d);
const pct=(x,d=1)=>x==null||isNaN(x)?'–':(100*x).toFixed(d)+'%';
const ci=c=>c?`[${pct(c[0],0)}–${pct(c[1],0)}]`:'';
const num=n=>n==null?'–':Number(n).toLocaleString();
const ago=t=>{if(!t)return 'never';const s=Date.now()/1000-t;return s<90?Math.round(s)+'s ago':s<5400?Math.round(s/60)+'m ago':s<172800?Math.round(s/3600)+'h ago':Math.round(s/86400)+'d ago';};
const badge=(l,c)=>l?`<span class="badge ${esc(c||l)}">${esc(l)}</span>`:'<span class="muted">–</span>';
const lcol=l=>`var(--${l})`;
async function api(p,opt){const r=await fetch('/api/'+p,opt);let j=null;try{j=await r.json();}catch(e){}if(!r.ok)throw new Error((j&&j.error)||r.status);return j;}
const pending=m=>`<div class="pending-box">${esc(m)}</div>`;

// ---- theme
(function(){let t=null;try{t=localStorage.getItem('fnp-theme');}catch(e){}
 if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
 document.documentElement.dataset.theme=t;
 $('#theme').onclick=()=>{t=t==='dark'?'light':'dark';document.documentElement.dataset.theme=t;try{localStorage.setItem('fnp-theme',t);}catch(e){}};})();

// ---- components
function probBars(probs,title){
  if(!probs)return `<div class="muted small">${esc(title)}: n/a</div>`;
  return `<div class="pbars"><div class="small"><b>${esc(title)}</b></div>`+LABELS.map(l=>{const p=probs[l]||0;
   return `<div class="r"><span>${l}</span><div class="bar"><i style="width:${p*100}%;background:${lcol(l)}"></i></div><span class="mono">${fmt(p,3)}</span></div>`;}).join('')+'</div>';
}
function stackBar(counts,total){ // counts {label:n}
  return `<div style="display:flex;height:14px;border-radius:4px;overflow:hidden;background:var(--line);width:100%">`+LABELS.map(l=>counts[l]?`<i title="${l}: ${counts[l]}" style="width:${100*counts[l]/total}%;background:${lcol(l)}"></i>`:'').join('')+'</div>';
}
const legend=()=>'<div class="legend">'+LABELS.map(l=>`<span><i style="background:${lcol(l)}"></i>${l}</span>`).join('')+'</div>';
function kvgrid(o){return '<div class="kv">'+Object.entries(o).map(([k,v])=>`<div><b>${esc(k)}</b>${typeof v==='number'?fmt(v,3):esc(v==null?'–':typeof v==='object'?JSON.stringify(v):v)}</div>`).join('')+'</div>';}
function flatTable(rows,cols){ // rows: [{name, ...scalars}]
  return `<div style="overflow:auto"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>`+rows.map(r=>'<tr>'+cols.map(c=>`<td>${typeof r[c]==='number'?fmt(r[c],4):esc(r[c]==null?'':r[c])}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function scalars(o){const r={};for(const[k,v]of Object.entries(o||{}))if(typeof v==='number'||typeof v==='string'||typeof v==='boolean')r[k]=v;return r;}

// ---- tabs
const TABS=[['overview','Overview'],['methods','Methods'],['data','Data'],['runs','Runs explorer'],['eval','Evaluation'],['gallery','Live gallery'],['verdicts','Verdicts'],['laya','Laya & RL'],['review','Review'],['try','Try it'],['annotate','Annotate']];
const R={};
function nav(){const h=(location.hash||'#overview').slice(1).split('?')[0];const t=TABS.find(x=>x[0]===h)?h:'overview';
 $('#tabs').innerHTML=TABS.map(([k,n])=>`<a class="${k===t?'on':''}" href="#${k}">${n}</a>`).join('');
 // Fresh container per navigation: a slow render from a previous tab writes into a
 // detached node and can no longer overwrite the tab the user switched to.
 const v=document.createElement('div');$('#view').replaceChildren(v);
 v.innerHTML=`<p class="muted"><span class="spin"></span>loading…</p>`;
 clearInterval(R.timer);
 R[t](v).catch(e=>{v.innerHTML=`<div class="card"><b>Error</b><p class="mono">${esc(e.message)}</p></div>`;});}
addEventListener('hashchange',nav);

// ---- Overview
R.overview=async function(v,fromTimer){
  const [o,u]=await Promise.all([api('overview'+(R.runTests?'?tests=1':'')),Promise.resolve(null)]);R.runTests=false;
  const S=Object.fromEntries(o.stages.map(s=>[s.id,s]));
  const us=S.units,L=(id,name,desc,st,ct)=>({id,name,desc,st,ct});
  const run=us.status;
  const nodes=[
   L('L0','Ingest','datasets, PTC, fact-check KB',S.data.status,num(S.data.count)+' items'),
   L('L1','Claims','Gemma claim extractor',run,us.count+' units'),
   L('L2','Propaganda','PTC span tagger (18 techniques)',run,'spans per unit'),
   L('L3','Evidence / KB','embeddings retrieve + Gemma stance',S.kb.status==='done'?run:S.kb.status,num(S.kb.count)+' KB entries'),
   L('L4','Independence','collapse duplicate origins',run,'n_eff'),
   L('L5','Reliability','priors + Beta posterior, truth discovery',run,'source reliability'),
   L('L6','Fusion maths','WoE, DS, meta-analysis, BP',run,'p_true, CI, ds_unc'),
   L('L7','Verdict','rule head + Gemma adjudicator + stacker',S.eval.status==='done'?'done':run,S.eval.status==='done'?'eval ready':'eval pending')];
  const pipe=nodes.map((n,i)=>`<div class="node ${n.st}"><div class="lv">${n.id} ${badge(n.st)}</div><div class="nm">${n.name}</div><div class="small muted">${n.desc}</div><div class="ct">${n.ct}</div></div>`+(i<nodes.length-1?'<div class="arrow">&rarr;</div>':'')).join('');
  const pctU=us.target?Math.min(100,100*us.count/us.target):0;
  const side=[S.laya,S.external].map(s=>`<div class="card"><div class="row"><b>${esc(s.name)}</b>${badge(s.status)}</div><div class="small muted">${esc(s.detail)}</div>${s.updated?`<div class="small muted">updated ${ago(s.updated)}</div>`:''}</div>`).join('');
  v.innerHTML=`<div class="card"><h2>Pipeline</h2><div class="pipe">${pipe}</div>
   <div style="margin-top:12px"><div class="row"><b>Units run</b><span class="mono">${us.count} / ${us.target}</span><span class="muted small">updated ${ago(us.updated)}</span></div><div class="bar"><i style="width:${pctU}%"></i></div></div></div>
   <div class="grid g3">${side}<div class="card"><div class="row"><b>Database</b>${badge(o.db.mode==='live'?'live':(o.db.mode||'n/a'),o.db.mode==='live'?'done':'pending')}</div>
     <div class="small mono">${Object.entries(o.db.tables||{}).map(([k,n])=>`${esc(k)}: ${num(n)}`).join('<br>')}</div></div></div>
   <div class="card"><h2>Stages</h2>${flatTable(o.stages.map(s=>({stage:s.name,status:s.status,detail:s.detail})),['stage','status','detail'])}</div>
   <div class="grid g2"><div class="card"><div class="row"><h2 style="margin:0">Tests</h2><button id="runtests">run pytest</button></div><div id="tests">${testsHtml(o.tests)}</div></div>
   <div class="card"><div class="row"><h2 style="margin:0">Logs</h2><button class="openlogs">open logs window</button></div><div class="small muted">${o.logs.map(l=>`${esc(l.name)} (${ago(l.mtime)})`).join(' &middot; ')||'none'}</div></div></div>
   <div class="card"><h2>Module inventory</h2><div class="row"><input id="modq" placeholder="filter modules"><span class="muted small">${o.modules.length} files</span></div><div id="mods"></div></div>`;
  $('#runtests').onclick=async e=>{e.target.disabled=true;$('#tests').innerHTML='<span class="spin"></span>running pytest (up to 120 s)…';
    try{const r=await api('overview?tests=1');$('#tests').innerHTML=testsHtml(r.tests);}catch(x){$('#tests').textContent=x.message;}e.target.disabled=false;};
  const drawMods=()=>{const q=($('#modq').value||'').toLowerCase();$('#mods').innerHTML=flatTable(o.modules.filter(m=>(m.path+m.doc).toLowerCase().includes(q)).map(m=>({path:m.path,lines:m.lines,description:m.doc})),['path','lines','description']);};
  $('#modq').oninput=drawMods;drawMods();
  v.querySelectorAll('.openlogs').forEach(b=>b.onclick=openLogs);
  if(!fromTimer)R.timer=setInterval(()=>{if(v.isConnected&&(location.hash||'#overview')==='#overview')R.overview(v,true).catch(()=>{});else clearInterval(R.timer);},20000);
};
function testsHtml(t){if(!t||!t.ran)return '<span class="muted">not run yet (click the button; takes a while)</span>';
  return `<div class="row">${badge(t.ok?'passing':'failing',t.ok?'done':'error')}<span class="mono small">${esc(t.summary)}</span><span class="muted small">${t.secs}s &middot; ${ago(t.at)}</span></div><pre class="log">${esc((t.tail||[]).join('\n'))}</pre>`;}

// ---- Methods
R.methods=async function(v){
  const [m,a]=await Promise.all([api('docs?path=docs/METHODS.md'),api('docs?path=docs/ARCHITECTURE.md')]);
  const r=mdRender(m.text,{toc:true});const tri=mdSection(a.text,/Practicality triage/i);
  v.innerHTML=`<div class="card md">${tri?mdRender(tri):''}</div><div class="layout"><nav class="toc card">${r.toc.map(t=>`<a class="l${t.n}" data-id="${t.id}">${esc(t.t)}</a>`).join('')}</nav><div class="card md">${r.html}</div></div>`;
  v.querySelectorAll('.toc a').forEach(a=>a.onclick=()=>document.getElementById(a.dataset.id)?.scrollIntoView({behavior:'smooth'}));
};

// ---- Data
R.data=async function(v){
  const d=await api('datasets');
  if(d.error)v.innerHTML=`<div class="card">${pending('dataset table unavailable: '+d.error)}</div>`;
  else{
    const splits=[...new Set(d.counts.map(r=>r.split))];
    const draw=sp=>{
      const by={};d.counts.filter(r=>!sp||r.split===sp).forEach(r=>{(by[r.dataset]=by[r.dataset]||{})[r.label5]=(by[r.dataset][r.label5]||0)+r.n;});
      const mx=Math.max(1,...Object.values(by).map(o=>Object.values(o).reduce((a,b)=>a+b,0)));
      $('#dsbars').innerHTML=Object.entries(by).map(([k,o])=>{const t=Object.values(o).reduce((a,b)=>a+b,0);
       return `<div style="margin:8px 0"><div class="row small" style="margin:0"><b>${esc(k)}</b><span class="muted">${num(t)}</span></div><div style="width:${100*t/mx}%;min-width:60px">${stackBar(o,t)}</div></div>`;}).join('');
      const rows=Object.entries(by).map(([k,o])=>({dataset:k,...Object.fromEntries(LABELS.map(l=>[l,o[l]||0])),total:Object.values(o).reduce((a,b)=>a+b,0)}));
      $('#dstab').innerHTML=flatTable(rows,['dataset',...LABELS,'total']);};
    v.innerHTML=`<div class="card"><div class="row"><h2 style="margin:0">Gold items (dataset &times; label5)</h2><select id="spsel"><option value="">all splits</option>${splits.map(s=>`<option>${esc(s)}</option>`).join('')}</select><span class="muted small">source: ${esc(d.mode)}</span></div>${legend()}<div id="dsbars"></div><div id="dstab" style="margin-top:12px"></div></div>
     <div class="card"><h2>Dataset &times; split &times; label5</h2><div style="max-height:360px;overflow:auto">${flatTable(d.counts,['dataset','split','label5','n'])}</div></div>
     <div class="grid g2"><div class="card"><h2>PTC propaganda spans by technique</h2><div id="ptc"></div></div><div class="card"><h2>Fact-check sites (Interface/factchecks)</h2><div id="fc"></div></div></div>`;
    $('#spsel').onchange=e=>draw(e.target.value);draw('');
    const bt=(d.ptc.by_technique||[]);const mxt=Math.max(1,...bt.map(x=>x.n));
    $('#ptc').innerHTML=bt.length?bt.map(x=>`<div class="pbars"><div class="r" style="grid-template-columns:200px 1fr 50px"><span>${esc(x.technique)}</span><div class="bar"><i style="width:${100*x.n/mxt}%;background:var(--PROPAGANDA)"></i></div><span class="mono">${x.n}</span></div></div>`).join(''):pending('no PTC spans');
    $('#fc').innerHTML=legend()+d.factchecks.map(f=>`<div style="margin:8px 0"><div class="row small" style="margin:0"><b>${esc(f.site)}</b><span class="muted">${num(f.n)}</span></div>${stackBar(f.by_verdict,f.n)}</div>`).join('');
  }
};

// ---- Runs explorer
R.runs=async function(v){
  const st=R.runsState=R.runsState||{offset:0,limit:50,q:'',dataset:'',split:'',label:'',pred:''};
  const load=async()=>{
    const qs=new URLSearchParams(Object.fromEntries(Object.entries(st).filter(([k,x])=>x!==''))).toString();
    const d=await api('units?'+qs);
    const opt=(arr,k,all)=>`<select data-k="${k}"><option value="">${all}</option>${arr.map(x=>`<option ${st[k]===x?'selected':''}>${esc(x)}</option>`).join('')}</select>`;
    v.innerHTML=`<div class="card"><div class="row">${opt(d.facets.dataset,'dataset','all datasets')}${opt(d.facets.split,'split','all splits')}${opt(d.facets.labels,'label','gold: any')}${opt(d.facets.labels,'pred','Gemma adj: any')}
     <input id="uq" placeholder="search claim text / id" value="${esc(st.q)}" style="min-width:240px"><button id="ugo">search</button>
     <span class="muted small">${num(d.total)} match &middot; ${num(d.all)} / ${d.target} units computed${d.all<d.target?' (run in progress)':''}</span></div>
     ${d.all?`<div style="overflow:auto"><table><thead><tr><th>claim</th><th>ds</th><th>gold</th><th>rule</th><th>Gemma adj</th><th>Gemma raw</th><th class="num">p_true [CI]</th><th class="num">ds_unc</th><th class="num">n_eff</th><th class="num">kb</th><th class="num">p_prop</th></tr></thead><tbody>`+
     d.rows.map(r=>`<tr class="click" data-id="${r.unit_id}"><td style="max-width:420px">${esc((r.claim||'').slice(0,160))}</td><td>${esc(r.dataset)}<div class="muted small">${esc(r.split)}</div></td><td>${badge(r.gold)}</td><td>${badge(r.rule)}</td><td>${badge(r.adj)}</td><td>${badge(r.raw)}</td>
      <td class="num mono">${fmt(r.p_true,2)} <span class="muted">[${fmt(r.p_true_lo,2)},${fmt(r.p_true_hi,2)}]</span></td><td class="num mono">${fmt(r.ds_uncertainty,2)}</td><td class="num mono">${fmt(r.n_eff,1)}</td><td class="num mono">${fmt(r.kb_match,2)}</td><td class="num mono">${fmt(r.p_prop,2)}</td></tr>`).join('')+`</tbody></table></div>
     <div class="row" style="margin-top:10px"><button id="prev" ${st.offset<=0?'disabled':''}>prev</button><button id="next" ${st.offset+st.limit>=d.total?'disabled':''}>next</button><span class="muted small">${d.total?st.offset+1:0}–${Math.min(d.total,st.offset+st.limit)} of ${num(d.total)}</span></div>`:pending('No units computed yet (data/runs/units_v1.jsonl absent or empty).')}</div>`;
    v.querySelectorAll('select[data-k]').forEach(s=>s.onchange=()=>{st[s.dataset.k]=s.value;st.offset=0;load();});
    const go=()=>{st.q=$('#uq').value;st.offset=0;load();};if($('#ugo')){$('#ugo').onclick=go;$('#uq').onkeydown=e=>{if(e.key==='Enter')go();};}
    if($('#prev')){$('#prev').onclick=()=>{st.offset=Math.max(0,st.offset-st.limit);load();};$('#next').onclick=()=>{st.offset+=st.limit;load();};}
    v.querySelectorAll('tr.click').forEach(tr=>tr.onclick=()=>openUnit(tr.dataset.id));
  };
  await load();
};
async function openUnit(id){
  const dr=$('#drawer');dr.classList.remove('hidden');$('#drawer-title').textContent='Unit '+id;$('#drawer-body').innerHTML='<span class="spin"></span>loading…';
  try{const u=await api('unit/'+encodeURIComponent(id));$('#drawer-body').innerHTML=unitHtml(u);}catch(e){$('#drawer-body').textContent=e.message;}
}
$('#drawer-close').onclick=()=>$('#drawer').classList.add('hidden');
addEventListener('keydown',e=>{if(e.key==='Escape')$('#drawer').classList.add('hidden');});
function evCards(ev){
  return (ev||[]).map(e=>`<div class="ev ${esc(e.stance)}"><div class="row" style="margin-bottom:4px">${badge(e.stance,e.stance)}<span class="small">${esc(e.source)}</span><span class="small muted">${esc(e.evidence_type||'')}</span>
   <span class="small mono">conf ${fmt(e.stance_conf,2)} &middot; rel ${fmt(e.reliability,2)} &middot; relevance ${fmt(e.relevance,2)} &middot; indep ${e.independence==null?'–':fmt(e.independence,2)} &middot; score ${fmt(e.retrieval_score,2)}</span></div>
   <div><span class="muted small">KB claim:</span> ${esc((e.text||'').slice(0,400))}</div>${e.url?`<div class="small"><a href="${esc(e.url)}" target="_blank" rel="noopener" style="color:var(--acc)">${esc(e.url.slice(0,90))}</a></div>`:''}<div class="muted small">origin ${esc(e.origin_key||'')}</div></div>`).join('')||'<div class="muted">no evidence retrieved</div>';
}
function unitHtml(u){
  const c=u.claim||{},f=u.features||{},adj=u.gemma_adj||{},sil=u.gemma_silver||{};
  const sp=(u.spans||[]);
  return `<div class="card"><div class="row">${badge(u.label5)}<span class="small muted">gold</span> ${badge(u.rule_argmax)}<span class="small muted">rule</span> ${badge(adj.label)}<span class="small muted">Gemma adj</span> ${badge(sil.label)}<span class="small muted">Gemma raw</span></div>
   <p>${esc(c.claim)}</p><div class="small muted">${esc(u.dataset)} / ${esc(u.split)} &middot; ${esc(u.item_id)} &middot; speaker ${esc(c.speaker||'–')} &middot; ${esc(c.claim_type||'')} &middot; lang ${esc(c.lang||'–')} &middot; check-worthy ${fmt(c.check_worthy,2)}</div></div>
   <div class="card"><h3>Probabilities</h3><div class="grid g2">${probBars(u.rule,'rule head')}${probBars(adj.probs,'Gemma adjudicator')}${probBars(sil.probs,'Gemma raw (silver)')}${probBars(u.laya&&u.laya.probs||u.laya_probs,'Laya')}</div></div>
   <div class="card"><h3>Rationale (Gemma adjudicator)</h3><p>${esc(adj.rationale||'–')}</p></div>
   <div class="card"><h3>Features</h3>${kvgrid(f)}</div>
   <div class="card"><h3>Evidence (${(u.evidence||[]).length})</h3>${evCards(u.evidence)}</div>
   <div class="card"><h3>Propaganda spans (${sp.length})</h3>${u.spans_summary?`<div class="small muted">techniques: ${esc((u.spans_summary.techniques||[]).join(', ')||'none')} &middot; one-sided ${fmt(u.spans_summary.one_sided,2)}</div>`:''}
    ${sp.map(s=>`<div class="ev"><span class="badge PROPAGANDA">${esc(s.technique)}</span> <span class="mono small">[${s.start}-${s.end}]</span> ${esc(s.text||s.span||'')}</div>`).join('')||'<div class="muted">no spans</div>'}</div>
   <div class="card"><h3>Laya state text</h3><pre class="log">${esc(u.state_text||'')}</pre></div>`;
}

// ---- Live gallery: newest-first suspects from the live scan, with unflag feedback for retraining
R.gallery=async function(v){
  const st=R.gst=R.gst||{show:'flagged',source:'',label:''};
  const sel=(k,opts)=>`<select data-k="${k}">${opts.map(([a,b])=>`<option value="${a}"${st[k]===a?' selected':''}>${b}</option>`).join('')}</select>`;
  v.innerHTML=`<div class="card"><h2>Live gallery <span class="muted small">newest first, refreshes every 10 s</span></h2><div class="row" id="gbar">
    ${sel('show',[['flagged','Flagged (suspects)'],['unflagged','Unflagged by you'],['all','All scored']])}
    ${sel('source',[['','social + news'],['social','social'],['news','news']])}
    ${sel('label',[['','any label'],...LABELS.map(l=>[l,l])])}<span id="gstat" class="muted small"></span></div></div><div id="glist"></div>`;
  v.querySelectorAll('#gbar select').forEach(e=>e.onchange=()=>{st[e.dataset.k]=e.value;load();});
  async function flag(id,action,label){await api('flag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,action,label})});load();}
  async function load(){
    const d=await api('gallery?'+new URLSearchParams(Object.fromEntries(Object.entries(st).filter(([k,x])=>x!==''))));
    const s=d.state||{},src=s.sources||{};
    $('#gstat').innerHTML=`${d.scanner_alive?'<span class="badge running">scanner running</span>':'<span class="badge stalled">scanner idle</span>'} scored ${num(d.scored)} of ${num(d.total_seen)} seen · ${num(d.matched)} match · ${num(d.n_flags)} feedback · social: ${esc(src.social||'–')} · news: ${esc(src.news||'–')}`;
    $('#glist').innerHTML=d.items.map(it=>{const f=it.flag,un=f&&(f.action==='unflag'||f.action==='relabel');
      return `<div class="card" style="${un?'opacity:.6':''}"><div class="row">${badge(it.label)}${it.plausibly_fake?'<span class="badge pf">plausibly fake</span>':''}
       <span class="mono">suspicion ${fmt(it.suspicion,2)}</span><span class="muted small">${esc(it.source)}/${esc(it.platform||'')} · posted ${esc((it.posted_at||'').slice(0,16))}</span>
       ${it.url?`<a href="${esc(it.url)}" target="_blank" rel="noopener noreferrer">link</a>`:''}
       ${f?`<span class="badge ${f.action==='confirm'?'LEGIT':'INCONCLUSIVE'}">${esc(f.action)}${f.label?': '+esc(f.label):''}</span>`:''}</div>
       <p style="white-space:pre-wrap;margin:8px 0">${esc(it.text.slice(0,500))}${it.text.length>500?'…':''}</p>
       ${it.claims.map(c=>`<div class="small">${badge(c.label)} ${esc(c.claim)}</div>`).join('')}
       <div class="small muted" style="margin:6px 0">${esc(it.rationale||'')}</div>
       <div class="row"><button class="g-un" data-id="${esc(it.id)}">Unflag – not suspicious</button><button class="g-ok" data-id="${esc(it.id)}">Confirm</button>
        <select class="g-rl" data-id="${esc(it.id)}"><option value="">relabel…</option>${LABELS.map(l=>`<option>${l}</option>`).join('')}</select>
        ${f?`<button class="g-clr" data-id="${esc(it.id)}">undo</button>`:''}</div></div>`;}).join('')||pending('nothing yet — the scanner appends items as they are scored');
    document.querySelectorAll('.g-un').forEach(b=>b.onclick=()=>flag(b.dataset.id,'unflag'));
    document.querySelectorAll('.g-ok').forEach(b=>b.onclick=()=>flag(b.dataset.id,'confirm'));
    document.querySelectorAll('.g-clr').forEach(b=>b.onclick=()=>flag(b.dataset.id,'clear'));
    document.querySelectorAll('.g-rl').forEach(b=>b.onchange=()=>b.value&&flag(b.dataset.id,'relabel',b.value));
  }
  await load();R.timer=setInterval(()=>load().catch(()=>{}),10000);
};

// ---- Evaluation
R.eval=async function(v){
  const d=await api('eval');const dv=d.derived;
  let h='';
  if(d.partial)h+=`<div class="card">${badge('partial','running')} derived metrics below use the ${num(dv.n)} units computed so far.</div>`;
  h+=`<div class="card"><h2>Report (reports/EVAL_v1.md)</h2>${d.markdown?`<div class="md">${mdRender(d.markdown)}</div>`:pending('pending — reports/EVAL_v1.md not yet written')}</div>`;
  const ej=d.eval_json;
  if(ej&&ej.results){const res=ej.results;let rows=[];
    if(Array.isArray(res))rows=res.map((r,i)=>({name:r.name||r.method||i,...scalars(r)}));
    else rows=Object.entries(res).map(([k,r])=>({name:k,...scalars(r)}));
    const cols=['name',...new Set(rows.flatMap(r=>Object.keys(r).filter(k=>k!=='name')))];
    h+=`<div class="card"><h2>Ablation / results (eval_v1.json)</h2>${flatTable(rows,cols)}</div>`;
    if(ej.breakdowns)h+=`<div class="card"><h2>Breakdowns (eval_v1.json)</h2><pre class="log">${esc(JSON.stringify(ej.breakdowns,null,1).slice(0,8000))}</pre></div>`;
  }else h+=`<div class="card"><h2>Ablation table</h2>${pending('pending — reports/eval_v1.json absent')}</div>`;
  h+=`<div class="card"><div class="row"><h2 style="margin:0">Derived from units</h2><select id="evsp"><option value="">all splits</option>${d.splits.map(s=>`<option>${esc(s)}</option>`).join('')}</select><span class="muted small">n=${num(dv.n)}; CIs are Wilson 95%</span></div>`;
  if(!dv.n)h+=pending('no units yet');
  else{
    h+=flatTable(Object.values(dv.methods).map(m=>({method:m.name,n:m.n,accuracy:pct(m.accuracy),CI:ci(m.acc_ci)})),['method','n','accuracy','CI']);
    for(const[k,m]of Object.entries(dv.methods)){
      h+=`<h3 style="margin-top:16px">${esc(m.name)}</h3><div class="grid g2"><div style="overflow:auto"><table><thead><tr><th>class</th><th class="num">support</th><th class="num">precision [CI]</th><th class="num">recall [CI]</th></tr></thead><tbody>`+
       LABELS.map(l=>{const c=m.per_class[l];return `<tr><td>${badge(l)}</td><td class="num">${c.support}</td><td class="num mono">${pct(c.precision)} <span class="muted">${ci(c.p_ci)}</span></td><td class="num mono">${pct(c.recall)} <span class="muted">${ci(c.r_ci)}</span></td></tr>`;}).join('')+`</tbody></table></div>`+confHtml(m.confusion)+'</div>';
    }
    h+=`<div class="grid g2" style="margin-top:16px"><div><h3>By dataset (accuracy)</h3>${bdTable(dv.by_dataset)}</div><div><h3>By kb_match</h3>${bdTable(dv.by_kb_match)}</div></div>`;
  }
  v.innerHTML=h+'</div>';
  const sel=$('#evsp');if(sel)sel.onchange=()=>evReload(v,sel.value);
};
async function evReload(v,sp){const d=await api('eval'+(sp?'?split='+sp:''));renderEvalDerived(v,d,sp);}
function renderEvalDerived(v,d,sp){
  const cards=v.querySelectorAll('.card');const last=cards[cards.length-1];const dv=d.derived;
  let h=`<div class="row"><h2 style="margin:0">Derived from units</h2><select id="evsp"><option value="">all splits</option>${d.splits.map(s=>`<option ${s===sp?'selected':''}>${esc(s)}</option>`).join('')}</select><span class="muted small">n=${num(dv.n)}; CIs are Wilson 95%</span></div>`;
  if(!dv.n)h+=pending('no units for this split');else{
    h+=flatTable(Object.values(dv.methods).map(m=>({method:m.name,n:m.n,accuracy:pct(m.accuracy),CI:ci(m.acc_ci)})),['method','n','accuracy','CI']);
    for(const m of Object.values(dv.methods)){
      h+=`<h3 style="margin-top:16px">${esc(m.name)}</h3><div class="grid g2"><div style="overflow:auto"><table><thead><tr><th>class</th><th class="num">support</th><th class="num">precision [CI]</th><th class="num">recall [CI]</th></tr></thead><tbody>`+LABELS.map(l=>{const c=m.per_class[l];return `<tr><td>${badge(l)}</td><td class="num">${c.support}</td><td class="num mono">${pct(c.precision)} <span class="muted">${ci(c.p_ci)}</span></td><td class="num mono">${pct(c.recall)} <span class="muted">${ci(c.r_ci)}</span></td></tr>`;}).join('')+`</tbody></table></div>`+confHtml(m.confusion)+'</div>';}
    h+=`<div class="grid g2" style="margin-top:16px"><div><h3>By dataset (accuracy)</h3>${bdTable(dv.by_dataset)}</div><div><h3>By kb_match</h3>${bdTable(dv.by_kb_match)}</div></div>`;}
  last.innerHTML=h;const s=$('#evsp');s.onchange=()=>evReload(v,s.value);
}
function confHtml(m){
  const mx=Math.max(1,...LABELS.flatMap(g=>LABELS.map(p=>m[g][p])));
  return `<div style="overflow:auto"><table class="cm"><thead><tr><th>gold \\ pred</th>${LABELS.map(l=>`<th class="c" style="text-align:center">${l.slice(0,5)}</th>`).join('')}</tr></thead><tbody>`+
   LABELS.map(g=>`<tr><td>${badge(g)}</td>`+LABELS.map(p=>{const n=m[g][p],a=n/mx;return `<td class="c" style="background:color-mix(in srgb,${g===p?'var(--LEGIT)':'var(--FAKE)'} ${Math.round(a*70)}%,transparent)">${n}</td>`;}).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function bdTable(o){return flatTable(Object.entries(o).map(([k,x])=>({group:k,n:x.n,rule:pct(x.rule.acc)+' '+ci(x.rule.ci),'Gemma adj':pct(x.adj.acc)+' '+ci(x.adj.ci),'Gemma raw':pct(x.raw.acc)+' '+ci(x.raw.ci)})),['group','n','rule','Gemma adj','Gemma raw']);}

// ---- Verdicts
R.verdicts=async function(v){
  const st=R.vState=R.vState||{offset:0,limit:50,q:'',dataset:'',label:'',plausibly_fake:''};
  const load=async()=>{
    const d=await api('verdicts?'+new URLSearchParams(Object.fromEntries(Object.entries(st).filter(([k,x])=>x!==''))));
    if(d.pending){v.innerHTML=`<div class="card"><h2>External document verdicts</h2>${pending('pending — data/runs/external_v1.jsonl and reports/verdict_v1_docs.jsonl not yet written')}</div>`;return;}
    v.innerHTML=`<div class="card"><div class="row"><select data-k="dataset"><option value="">all datasets</option>${d.facets.dataset.map(x=>`<option ${st.dataset===x?'selected':''}>${esc(x)}</option>`).join('')}</select>
     <select data-k="label"><option value="">any verdict</option>${d.facets.labels.map(x=>`<option ${st.label===x?'selected':''}>${esc(x)}</option>`).join('')}</select>
     <label><input type="checkbox" id="pfonly" ${st.plausibly_fake?'checked':''}> plausibly fake only</label><input id="vq" placeholder="search id / claim" value="${esc(st.q)}"><button id="vgo">search</button><span class="muted small">${num(d.total)} of ${num(d.all)} docs, ranked by P(FAKE)+P(MISLEADING)</span></div>
     <div style="overflow:auto"><table><thead><tr><th>#</th><th>doc</th><th>dataset</th><th>verdict</th><th class="num">P(FAKE)+P(MISL)</th><th>probs</th><th>claims</th></tr></thead><tbody>`+
     d.rows.map((r,i)=>`<tr><td class="muted">${st.offset+i+1}</td><td class="mono small" style="max-width:260px;word-break:break-all">${esc(r.id)}</td><td>${esc(r.dataset)}</td><td>${badge(r.label)} ${r.plausibly_fake?'<span class="badge pf">plausibly fake</span>':''}</td>
      <td class="num mono">${fmt(r.score,3)}</td><td style="min-width:160px">${stackBar(r.probs,Object.values(r.probs).reduce((a,b)=>a+b,0)||1)}</td>
      <td class="small">${(r.claims||[]).slice(0,3).map(c=>`${badge(c.verdict&&c.verdict.label)} ${esc((c.claim||'').slice(0,110))}`).join('<br>')}${(r.claims||[]).length>3?`<br><span class="muted">+${r.claims.length-3} more</span>`:''}</td></tr>`).join('')+`</tbody></table></div>
     <div class="row" style="margin-top:10px"><button id="vprev" ${st.offset<=0?'disabled':''}>prev</button><button id="vnext" ${st.offset+st.limit>=d.total?'disabled':''}>next</button></div></div>`;
    v.querySelectorAll('select[data-k]').forEach(s=>s.onchange=()=>{st[s.dataset.k]=s.value;st.offset=0;load();});
    $('#pfonly').onchange=e=>{st.plausibly_fake=e.target.checked?'1':'';st.offset=0;load();};
    const go=()=>{st.q=$('#vq').value;st.offset=0;load();};$('#vgo').onclick=go;$('#vq').onkeydown=e=>{if(e.key==='Enter')go();};
    $('#vprev').onclick=()=>{st.offset=Math.max(0,st.offset-st.limit);load();};$('#vnext').onclick=()=>{st.offset+=st.limit;load();};
  };
  await load();
};

// ---- Laya & RL
R.laya=async function(v){
  const d=await api('laya');
  let h='';
  if(!d.cards.length)h+=pending('no Laya models found (data/models/laya_fnp_v*)');
  for(const c of d.cards){const k=c.card;
    if(!k){h+=`<div class="card"><h2>${esc(c.name)}</h2>${pending('training_card.json missing')}</div>`;continue;}
    const ev=k.eval||k.eval_verdict||{};
    h+=`<div class="card"><div class="row"><h2 style="margin:0">${esc(c.name)}</h2>${badge('stage '+k.stage,'running')}<span class="muted small">created ${esc(k.created)} &middot; parent ${esc(k.parent||'–')} &middot; weights ${c.has_weights?'present':'absent'}</span></div>
     <div class="kv">${[['base',k.base||(k.config||{}).base],['wall (s)',k.wall_s],['GPU peak (GB)',k.gpu_peak_alloc_gb],['s / microbatch',k.step_time_s_per_microbatch],['frozen params',k.frozen_params],['OOM retries',k.oom_retries],['train rows',(k.counts||{}).train_rows],['calib rows',(k.counts||{}).calib_rows],['eval rows',(k.counts||{}).eval_rows],['temperature',k.temperature]].filter(x=>x[1]!=null).map(([a,b])=>`<div><b>${a}</b>${typeof b==='number'?fmt(b,3):esc(Array.isArray(b)?b.map(x=>fmt(x,3)).join(', '):b)}</div>`).join('')}</div>
     <div class="grid g2" style="margin-top:10px">
      ${[['eval (calibrated)',k.eval],['eval (uncalibrated)',k.eval_uncalibrated],['eval verdict',k.eval_verdict],['parent uncalibrated',k.eval_parent_uncalibrated],['base zero-shot',k.eval_base_zero_shot]].filter(x=>x[1]).map(([t,o])=>`<div><h3>${t}</h3>${kvgrid(scalars(o))}</div>`).join('')}
     </div>
     ${k.epoch_loss||k.epoch_stats?`<h3>Epochs</h3><pre class="log">${esc(JSON.stringify(k.epoch_stats||k.epoch_loss,null,1))}</pre>`:''}
     ${k.abstention_thresholds?`<h3>Abstention thresholds</h3>${kvgrid(k.abstention_thresholds)}`:''}
     ${k.calibration?`<h3>Calibration</h3><pre class="log">${esc(JSON.stringify(k.calibration,null,1).slice(0,3000))}</pre>`:''}
     ${k.assumptions?`<h3>Assumptions</h3><pre class="log">${esc(JSON.stringify(k.assumptions,null,1))}</pre>`:''}
     ${c.rl_agent?`<h3>RL agent config</h3>${kvgrid(scalars(c.rl_agent))}${c.rl_agent.training?`<h3>RL training</h3>${kvgrid(scalars(c.rl_agent.training))}`:''}`:''}
     <details><summary class="muted small">full training_card.json</summary><pre class="log">${esc(JSON.stringify(k,null,1))}</pre></details></div>`;}
  h+=`<div class="card"><h2>Routing bandit</h2>${d.bandit.length?d.bandit.map(b=>`<div><b class="mono">${esc(b.path)}</b><pre class="log">${esc(b.data?JSON.stringify(b.data,null,1).slice(0,4000):b.text||'')}</pre></div>`).join(''):pending('no routing-bandit output found yet. Implementation: '+d.rl_modules.join(', '))}</div>`;
  v.innerHTML=h;
};

// ---- Review
R.review=async function(v){
  const l=await api('docs');const groups={};l.docs.forEach(x=>(groups[x.dir]=groups[x.dir]||[]).push(x));
  const sel=R.revSel||(groups['docs/review']||[])[0]?.path;
  v.innerHTML=`<div class="layout"><div class="toc card">${Object.entries(groups).map(([g,xs])=>`<div class="muted small" style="margin-top:8px">${esc(g)}/</div>`+xs.map(x=>`<a data-p="${esc(x.path)}" ${x.path===sel?'style="color:var(--acc);font-weight:600"':''}>${esc(x.name)}</a>`).join('')).join('')}</div><div class="card md" id="docview"></div></div>`;
  const show=async p=>{R.revSel=p;$('#docview').innerHTML='<span class="spin"></span>';const d=await api('docs?path='+encodeURIComponent(p));
    $('#docview').innerHTML=`<div class="muted small mono">${esc(p)}</div>`+mdRender(d.text);v.querySelectorAll('.toc a').forEach(a=>a.style.cssText=a.dataset.p===p?'color:var(--acc);font-weight:600':'');};
  v.querySelectorAll('.toc a').forEach(a=>a.onclick=()=>show(a.dataset.p));
  if(sel)show(sel);else $('#docview').innerHTML=pending('no review docs');
};

// ---- Annotate (the tool is its own server: scripts/29_annot_server.py, port 8890; this tab links to it)
R.annotate=async function(v){
  const url=`${location.protocol}//${location.hostname}:8890/`;
  v.innerHTML=`<div class="card"><h2>Annotation tool</h2>
  <p class="muted">Human labelling of <b>claims</b> (span, form of claim, check-worthiness) and <b>propaganda</b> (span, one or more of the 14 PTC techniques) on items pulled incrementally from the news and social databases. Labels are stored in the dedicated PostgreSQL annotation database and exported as versioned training sets.</p>
  <p><a href="${url}" target="_blank" rel="noopener"><button class="primary">Open the annotation tool</button></a> <span class="muted small mono">${esc(url)}</span></p>
  <p class="muted small">Not running? <code>.venv/bin/python scripts/29_annot_server.py --port 8890</code> (first admin: <code>scripts/31_annot_users.py create NAME --role admin</code>). Guide: <code>docs/ANNOTATION.md</code>.</p></div>`;
};

// ---- Try it
R.try=async function(v){
  v.innerHTML=`<div class="card"><h2>Analyse a claim or article live</h2><p class="muted small">Runs <code>fnp.pipeline.analyse_text</code> with the Gemma endpoint; typically about 15 s for a short claim.</p>
   <textarea id="pt" rows="7" style="width:100%" placeholder="Paste a claim, post or article…"></textarea><div class="row" style="margin-top:8px"><button class="primary" id="pgo">Analyse</button><span id="pst" class="muted"></span></div></div><div id="pres"></div>`;
  $('#pgo').onclick=async()=>{const t=$('#pt').value.trim();if(!t)return;const b=$('#pgo');b.disabled=true;$('#pst').innerHTML='<span class="spin"></span>running (Gemma)…';$('#pres').innerHTML='';
    try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);
      $('#pst').textContent=`done in ${j.secs}s`;$('#pres').innerHTML=predHtml(j);}catch(e){$('#pst').textContent='';$('#pres').innerHTML=`<div class="card"><b>Error</b><p class="mono">${esc(e.message)}</p></div>`;}b.disabled=false;};
};
function predHtml(j){
  const d=j.doc||{};
  return `<div class="card"><h2>Document verdict</h2><div class="row">${badge(d.label)}${d.plausibly_fake?'<span class="badge pf">plausibly fake</span>':''}<span class="muted small">${esc(d.decider||'')}</span></div>${probBars(d.probs,'doc probabilities')}${d.rationale?`<p>${esc(d.rationale)}</p>`:''}</div>`+
  (j.claims||[]).map(c=>{const vd=c.verdict||{};return `<div class="card"><div class="row">${badge(vd.label)}${vd.plausibly_fake?'<span class="badge pf">plausibly fake</span>':''}<b>${esc((c.claim&&c.claim.claim)||'')}</b></div>
    <div class="grid g2">${probBars(vd.probs,'final')}${probBars(c.rule,'rule head')}${probBars(c.gemma_adj&&c.gemma_adj.probs,'Gemma adjudicator')}${probBars(c.gemma_silver&&c.gemma_silver.probs,'Gemma raw')}</div>
    <p>${esc(vd.rationale||(c.gemma_adj&&c.gemma_adj.rationale)||'')}</p><h3>Features</h3>${kvgrid(c.features||{})}<h3>Evidence</h3>${evCards(c.evidence)}</div>`;}).join('')+
  `<div class="card"><h3>Propaganda spans</h3>${(j.spans||[]).map(s=>`<div class="ev"><span class="badge PROPAGANDA">${esc(s.technique)}</span> ${esc(s.text||'')}</div>`).join('')||'<div class="muted">none</div>'}</div>`;
}

// live header badge
async function tick(){try{const o=await api('overview');const u=o.stages.find(s=>s.id==='units');$('#live').textContent=`units ${u.count}/${u.target}`;}catch(e){$('#live').textContent='';}}
function openLogs(){window.open('logs.html','fnp-logs','width=1000,height=720');}
$('#logsbtn').onclick=openLogs;
nav();tick();setInterval(tick,30000);
