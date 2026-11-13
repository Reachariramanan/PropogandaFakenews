// Minimal markdown renderer: headings, tables, fenced code, lists, blockquotes, links, bold/italic/inline code.
(function(){
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const slug=s=>s.toLowerCase().replace(/<[^>]+>/g,'').replace(/[^\wÀ-￿]+/g,'-').replace(/^-|-$/g,'');
function inline(s){
  const codes=[];
  s=s.replace(/`([^`]+)`/g,(m,c)=>{codes.push(c);return '\u0000'+(codes.length-1)+'\u0000';});
  s=esc(s);
  s=s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,(m,t,u)=>/^(https?:|#|\/)/.test(u)?`<a href="${u}" target="_blank" rel="noopener">${t}</a>`:t);
  s=s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[\s(])\*([^*\s][^*]*)\*/g,'$1<i>$2</i>');
  return s.replace(/\u0000(\d+)\u0000/g,(m,i)=>'<code>'+esc(codes[i])+'</code>');
}
function cells(l){return l.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());}
window.mdRender=function(src,opts){
  const L=String(src||'').replace(/\r/g,'').split('\n'),out=[],toc=[];let i=0;
  while(i<L.length){
    let l=L[i];
    if(/^```/.test(l)){const b=[];i++;while(i<L.length&&!/^```/.test(L[i]))b.push(L[i++]);i++;out.push('<pre><code>'+esc(b.join('\n'))+'</code></pre>');continue;}
    let m=/^(#{1,6})\s+(.*)$/.exec(l);
    if(m){const n=m[1].length,t=m[2],id='h-'+slug(t);if(n>=2&&n<=3)toc.push({n,t:t.replace(/[`*]/g,''),id});out.push(`<h${n} id="${id}">${inline(t)}</h${n}>`);i++;continue;}
    if(/^\s*\|/.test(l)&&i+1<L.length&&/^\s*\|?\s*:?-{2,}/.test(L[i+1])){
      const h=cells(l);i+=2;const rows=[];while(i<L.length&&/^\s*\|/.test(L[i]))rows.push(cells(L[i++]));
      out.push('<table><thead><tr>'+h.map(c=>'<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+inline(c)+'</td>').join('')+'</tr>').join('')+'</tbody></table>');continue;}
    if(/^\s*([-*+]|\d+\.)\s+/.test(l)){
      const ord=/^\s*\d+\./.test(l),items=[];
      while(i<L.length&&/^\s*([-*+]|\d+\.)\s+/.test(L[i])){let t=L[i].replace(/^\s*([-*+]|\d+\.)\s+/,'');i++;
        while(i<L.length&&/^\s{2,}\S/.test(L[i])&&!/^\s*([-*+]|\d+\.)\s+/.test(L[i]))t+=' '+L[i++].trim();items.push(t);}
      out.push((ord?'<ol>':'<ul>')+items.map(t=>'<li>'+inline(t)+'</li>').join('')+(ord?'</ol>':'</ul>'));continue;}
    if(/^>\s?/.test(l)){const b=[];while(i<L.length&&/^>\s?/.test(L[i]))b.push(L[i++].replace(/^>\s?/,''));out.push('<blockquote>'+inline(b.join(' '))+'</blockquote>');continue;}
    if(/^\s*---+\s*$/.test(l)){out.push('<hr>');i++;continue;}
    if(!l.trim()){i++;continue;}
    const p=[];while(i<L.length&&L[i].trim()&&!/^(#{1,6}\s|```|\s*\||\s*([-*+]|\d+\.)\s|>)/.test(L[i]))p.push(L[i++]);
    if(!p.length){p.push(L[i++]);}
    out.push('<p>'+inline(p.join(' '))+'</p>');
  }
  const html=out.join('\n');
  return opts&&opts.toc?{html,toc}:html;
};
window.mdSection=function(src,titleRe){ // extract a ## section
  const L=String(src||'').split('\n');let s=-1;
  for(let i=0;i<L.length;i++){if(/^##\s/.test(L[i])&&titleRe.test(L[i])){s=i;break;}}
  if(s<0)return '';let e=L.length;for(let i=s+1;i<L.length;i++)if(/^##\s/.test(L[i])){e=i;break;}
  return L.slice(s,e).join('\n');
};
})();
