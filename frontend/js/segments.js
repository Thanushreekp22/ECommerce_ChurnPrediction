function segmentRows(segs){
  return Object.entries(segs).flatMap(([column,values])=>Object.entries(values).map(([value,stats])=>({column,value,...stats,rate:(stats.churned||0)/Math.max(stats.total||0,1)})));
}
function renderSegmentOverview(data){
  const all=segmentRows(data.segments||{}), summary=data.summary||{};
  const top=all.filter(x=>x.total>=3).sort((a,b)=>b.rate-a.rate)[0]||all.sort((a,b)=>b.rate-a.rate)[0];
  const html=top?`<section class="card segment-highlight"><div class="segment-eyebrow">Priority segment</div><h2>${escapeHtml(top.value)}</h2><p>${escapeHtml(top.column)} · ${(top.rate*100).toFixed(1)}% predicted churn across ${top.total} customers</p></section>`:
    `<section class="card segment-highlight"><div class="segment-eyebrow">Customer segments</div><h2>Awaiting segment data</h2><p>Upload data containing categorical customer attributes to compare groups.</p></section>`;
  document.getElementById('segment-overview').innerHTML=html+
    `<section class="card segment-stat danger"><span>High risk</span><strong>${Number(summary.high_risk||0).toLocaleString()}</strong><p class="muted">customers needing attention</p></section>`+
    `<section class="card segment-stat good"><span>Stable</span><strong>${Number(summary.low_risk||0).toLocaleString()}</strong><p class="muted">low-risk customers</p></section>`;
}
function renderSegmentCards(data){
  const segs=data.segments||{}, grid=document.getElementById('segment-grid');
  const entries=Object.entries(segs);
  if(!entries.length){grid.innerHTML='<div class="card empty">No categorical segments are available for this dataset.</div>';return;}
  grid.innerHTML=entries.slice(0,12).map(([col,values])=>{
    const rows=Object.entries(values).sort((a,b)=>(b[1].total||0)-(a[1].total||0)).slice(0,8);
    const totalCustomers=rows.reduce((sum,[,c])=>sum+(c.total||0),0);
    return `<section class="card segment-card"><div class="segment-card-head"><div><h3>${escapeHtml(col)}</h3><p>Compare predicted churn by group</p></div><span class="segment-count">${totalCustomers} customers</span></div><div class="segment-card-body">${rows.map(([value,c])=>{
      const total=c.total||1,churn=(c.churned||0)/total*100,retained=(c.retained||0)/total*100;
      return `<div class="segment-item"><div class="segment-label"><span title="${escapeHtml(value)}">${escapeHtml(value)}</span><span>${churn.toFixed(0)}% churn</span></div><div class="segment-track"><div class="segment-churned" style="width:${churn}%"></div><div class="segment-retained" style="width:${retained}%"></div></div></div>`;
    }).join('')}</div></section>`;
  }).join('');
}
async function initSegments(){
  const state = await requireAnalytics();  if (!state) return;
  basePage('segments',`
    <div class="page-head"><div><h1>Customer Segments</h1><p>Compare churn behaviour across categorical customer groups.</p></div></div>
    <div class="segment-overview" id="segment-overview"></div>
    <div class="legend-small"><span><i style="background:var(--red)"></i>Churned / predicted churn</span><span><i style="background:var(--green)"></i>Retained / stable</span></div>
    <div class="segment-grid" id="segment-grid"></div>
  `);
  renderSegmentOverview(state.payload);renderSegmentCards(state.payload);
}
document.addEventListener('DOMContentLoaded',initSegments);
