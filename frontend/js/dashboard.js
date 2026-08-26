let state = null;
let data = null;
let customers = [];

function timelineData() {
  const t = data.timeline || {};
  return [
    {label:'1 Month', value:t['1_month'] ?? t['1 month'] ?? 0},
    {label:'3 Months', value:t['3_months'] ?? t['3 months'] ?? 0},
    {label:'6 Months', value:t['6_months'] ?? t['6 months'] ?? 0},
    {label:'Stable', value:t.stable ?? 0}
  ];
}
function riskData() {
  const s = data.summary || {};
  return [
    {label:'High',value:s.high_risk||0},
    {label:'Medium',value:s.medium_risk||0},
    {label:'Low',value:s.low_risk||0}
  ];
}
const timelineColors=['#c83b45','#b98900','#376ca8','#168a57'];
const riskColors=['#c83b45','#b98900','#168a57'];

function renderDonut(id, items, colors) {
  const svg=document.getElementById(id);
  if(!svg)return;
  const total=items.reduce((a,b)=>a+b.value,0)||1, cx=60,cy=60,r=44,stroke=15;
  let offset=0, html='';
  items.forEach((item,i)=>{
    const pct=item.value/total, dash=pct*2*Math.PI*r, gap=2*Math.PI*r-dash;
    html += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${colors[i]}"
      stroke-width="${stroke}" stroke-dasharray="${dash} ${gap}"
      transform="rotate(${offset*360-90} ${cx} ${cy})" opacity=".92"/>`;
    offset += pct;
  });
  html += `<text x="60" y="60" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="800" fill="#16202b">${total.toLocaleString()}</text>`;
  svg.innerHTML=html;
}

function renderLegend(id, items, colors, listFn) {
  const el=document.getElementById(id);
  el.innerHTML=items.map((x,i)=>`
    <div class="legend-item" data-label="${escapeHtml(x.label)}">
      <span class="legend-dot" style="background:${colors[i]}"></span>
      <span class="legend-label">${escapeHtml(x.label)}</span>
      <span class="legend-value">${x.value.toLocaleString()}</span>
    </div>`).join('');
  if(listFn) el.querySelectorAll('.legend-item').forEach((node,i)=>node.onclick=()=>listFn(items[i].label));
}

function renderKPIs() {
  const s=data.summary||{};
  document.getElementById('kpis').innerHTML=`
    <div class="kpi-card kpi-info"><div class="kpi-label">Total Customers</div><div class="kpi-value">${(s.total_customers||0).toLocaleString()}</div><div class="kpi-sub">in dataset</div></div>
    <div class="kpi-card kpi-danger"><div class="kpi-label">Predicted to Churn</div><div class="kpi-value">${(s.predicted_churn||0).toLocaleString()}</div><div class="kpi-sub">${s.churn_rate_pct ?? 0}% of customers</div></div>
    <div class="kpi-card kpi-warn"><div class="kpi-label">At Risk</div><div class="kpi-value">${(s.high_risk||0).toLocaleString()}</div><div class="kpi-sub">high risk customers</div></div>
    <div class="kpi-card kpi-good"><div class="kpi-label">Stable</div><div class="kpi-value">${(s.low_risk||0).toLocaleString()}</div><div class="kpi-sub">low risk customers</div></div>`;
}

/* ── Human-readable feature names ─────────────────────────────────────────── */
const FEATURE_LABELS={
  'Tenure':'Tenure (months)','tenure_months':'Tenure (months)','tenure_days':'Days since signup',
  'account_age_days':'Account age (days)',
  'CashbackAmount':'Average cashback','cashback_amount':'Average cashback',
  'Complain':'Raised a complaint','Complain_1':'Raised a complaint','Complain_0':'No complaints',
  'WarehouseToHome':'Distance to warehouse','HourSpendOnApp':'Hours on app per day',
  'NumberOfAddress':'Saved addresses','NumberOfDeviceRegistered':'Registered devices',
  'OrderAmountHikeFromlastYear':'Order growth vs last year','OrderCount':'Orders last month',
  'DaySinceLastOrder':'Days since last order','CouponUsed':'Coupons used',
  'SatisfactionScore':'Satisfaction score','CityTier':'City tier','age':'Customer age',
  'total_orders':'Total orders','total_spend':'Total spend','average_order_value':'Average order value',
  'recency':'Recency (days)','wishlist_items':'Wishlist items','returns_count':'Returns',
  'monthly_spend':'Monthly spend','support_calls':'Support calls',
  'RFM_Composite_Score':'RFM composite','RFM_Recency_Score':'RFM recency','RFM_Frequency_Score':'RFM frequency',
  'MonthlyCharges':'Monthly charges','TotalCharges':'Total charges'
};
function prettyFeature(name){
  const direct=FEATURE_LABELS[name];
  if(direct)return direct;
  const idx=name.indexOf('_');
  if(idx>0){
    const base=name.slice(0,idx),val=name.slice(idx+1);
    const bl=FEATURE_LABELS[base];
    return `${(bl||base.replace(/_/g,' '))}: ${val.replace(/_/g,' ')}`;
  }
  return name.replace(/_/g,' ');
}
function pearsonSign(xs,ys){
  const n=xs.length;if(n<5)return 0;
  const mx=xs.reduce((a,b)=>a+b,0)/n,my=ys.reduce((a,b)=>a+b,0)/n;
  let sxy=0,sxx=0,syy=0;
  for(let i=0;i<n;i++){const dx=xs[i]-mx,dy=ys[i]-my;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy;}
  if(sxx<=0||syy<=0)return 0;
  const r=sxy/Math.sqrt(sxx*syy);
  return r>0.15?1:(r<-0.15?-1:0);
}
function driverDirection(feature){
  const xs=[],ys=[];
  customers.forEach(c=>{
    const v=c[feature];if(v==null)return;
    const num=typeof v==='number'?v:parseFloat(v);
    if(!Number.isNaN(num)){xs.push(num);ys.push(c.churn_probability||0);}
  });
  return pearsonSign(xs,ys);
}
function renderDrivers() {
  const drivers=(data.top_drivers||[]).slice(0,8);
  const max=drivers[0]?.importance||1;
  document.getElementById('drivers').innerHTML=drivers.length?drivers.map(d=>{
    const sign=driverDirection(d.feature);
    const cls=sign>0?'up':(sign<0?'down':'');
    const arrow=sign>0?'▲ raises risk':(sign<0?'▼ lowers risk':'influences churn');
    const pct=Math.max(4,(d.importance/max*100)).toFixed(0);
    return `<div class="bar-item">
      <div class="bar-label"><span title="${escapeHtml(d.feature)}">${escapeHtml(prettyFeature(d.feature))} <span class="dir ${cls||'flat'}">${arrow}</span></span><span class="driver-impact">${pct}%</span></div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('')
  +'<div class="driver-legend"><span class="dir up">▲</span> pushes churn risk up&nbsp;&nbsp;<span class="dir down">▼</span> reduces it&nbsp;&nbsp;·&nbsp;&nbsp;% = influence vs strongest driver</div>'
  :'<div class="empty">No churn drivers returned by the model.</div>';
}

function histBandColor(binStart){
  const mid=(binStart+0.025)*100;
  if(mid>=70)return '#c83b45';
  if(mid>=40)return '#b98900';
  return '#168a57';
}
function renderHistogram() {
  const hist=data.histogram||[], el=document.getElementById('histogram');
  if(!hist.length){el.innerHTML='<div class="empty">Probability distribution unavailable.</div>';return;}
  const W=460,H=196,padL=36,padR=6,padT=16,padB=28;
  const plotW=W-padL-padR,plotH=H-padT-padB;
  const max=Math.max(...hist.map(h=>h.count||0),1);
  const niceMax=Math.max(4,Math.ceil(max/4)*4);
  const bw=plotW/hist.length;
  const yOf=v=>padT+plotH-(v/niceMax)*plotH;

  let svg=`<svg viewBox="0 0 ${W} ${H}" class="hist-svg" role="img" aria-label="Number of customers by churn probability">`;
  [0,0.25,0.5,0.75,1].forEach(f=>{
    const v=niceMax*f,yv=yOf(v);
    svg+=`<line x1="${padL}" y1="${yv}" x2="${W-padR}" y2="${yv}" stroke="#e6edf3"/>`;
    svg+=`<text x="${padL-6}" y="${yv+3.5}" text-anchor="end" font-size="9.5" fill="#64748b">${Math.round(v)}</text>`;
  });
  hist.forEach((h,i)=>{
    const v=h.count||0,x=padL+i*bw+1,w=Math.max(bw-2,2);
    const yTop=yOf(v),height=Math.max(plotH+padT-yTop,v>0?2:0);
    svg+=`<rect class="hist-col" x="${x.toFixed(2)}" y="${yTop.toFixed(2)}" width="${w.toFixed(2)}" height="${height.toFixed(2)}" rx="2" fill="${histBandColor(h.bin_start)}" opacity=".9"><title>${(h.bin_start*100).toFixed(0)}–${(h.bin_end*100).toFixed(0)}% chance · ${v.toLocaleString()} customer${v===1?'':'s'}</title></rect>`;
  });
  const tx=padL+plotW*0.5;
  svg+=`<line x1="${tx}" y1="${padT}" x2="${tx}" y2="${padT+plotH}" stroke="#16202b" stroke-width="1.3" stroke-dasharray="4 3" opacity=".55"/>`;
  svg+=`<text x="${tx}" y="${padT-5}" text-anchor="middle" font-size="9" font-weight="700" fill="#16202b" opacity=".65">50% cutoff</text>`;
  [[0,'0%'],[0.25,'25%'],[0.5,'50%'],[0.75,'75%'],[1,'100%']].forEach(([f,lbl])=>{
    svg+=`<text x="${padL+plotW*f}" y="${H-9}" text-anchor="middle" font-size="9.5" fill="#64748b">${lbl}</text>`;
  });
  svg+=`<text x="${W-padR}" y="${H-9}" text-anchor="end" font-size="9" fill="#94a3b8">churn probability →</text>`;
  svg+='</svg>';

  const stay=hist.filter(h=>h.bin_start<0.5).reduce((a,b)=>a+(b.count||0),0);
  const gone=hist.filter(h=>h.bin_start>=0.5).reduce((a,b)=>a+(b.count||0),0);
  el.innerHTML=`${svg}
    <div class="hist-summary">
      <span class="hist-pill stay"><i></i>${stay.toLocaleString()} likely to stay</span>
      <span class="hist-pill churn"><i></i>${gone.toLocaleString()} likely to churn</span>
    </div>
    <div class="hist-hint">Each bar counts customers in that chance range. Bars right of the dashed 50% line are predicted to churn.</div>`;
}
function renderHighRiskCustomers(){
  const list=customers.slice().sort((a,b)=>(b.churn_probability||0)-(a.churn_probability||0)).slice(0,5);
  const el=document.getElementById('high-risk-list');
  el.innerHTML=list.length?list.map(c=>`<tr><td class="mono">${escapeHtml(c.customer_id)}</td><td><b>${((c.churn_probability||0)*100).toFixed(0)}%</b></td><td><span class="risk-badge ${escapeHtml(c.risk_level||'Medium')}">${escapeHtml(c.risk_level||'Medium')}</span></td><td><button class="text-action" data-id="${escapeHtml(c.customer_id)}">View</button></td></tr>`).join(''):'<tr><td colspan="4" class="empty">No customers returned by this analysis.</td></tr>';
  el.querySelectorAll('[data-id]').forEach(button=>button.onclick=async()=>{await saveCustomerFocus(button.dataset.id);go('customers.html');});
}

function showSegmentList(label) {
  const found=customers.filter(c=>{
    const tl=String(c.churn_timeline||'').toLowerCase(), rl=String(c.risk_level||'').toLowerCase();
    const x=label.toLowerCase();
    if(x==='1 month')return tl.includes('1');
    if(x==='3 months')return tl.includes('3');
    if(x==='6 months')return tl.includes('6');
    if(x==='stable')return tl.includes('stable');
    return rl===x;
  });
  document.getElementById('list-title').textContent=`${label} (${found.length})`;
  document.getElementById('list-body').innerHTML=found.slice().sort((a,b)=>b.churn_probability-a.churn_probability).slice(0,100).map(c=>`
    <div class="rec-card" style="justify-content:space-between;cursor:pointer" data-id="${escapeHtml(c.customer_id)}">
      <span class="mono">${escapeHtml(c.customer_id)}</span>
      <span><span class="risk-badge ${c.risk_level}">${escapeHtml(c.risk_level)}</span> <span class="mono">${(c.churn_probability*100).toFixed(1)}%</span></span>
    </div>`).join('') || '<div class="empty">No customers in this segment.</div>';
  document.querySelectorAll('#list-body [data-id]').forEach(x=>x.onclick=async()=>{
    try {
      await saveCustomerFocus(x.dataset.id);
      go('customers.html');
    } catch (error) {
      console.error('Could not save customer focus:', error);
      toast('Unable to open customer details.');
    }
  });
  showModal('list-modal');
}

function renderBanner() {
  const area=document.getElementById('banner');
  if(data.dataset_type==='unlabeled' && data.model){
    const model=data.model;
    area.innerHTML=`<div class="banner"><div><strong>Prediction completed</strong><span> · ${escapeHtml(data.source_file||'Uploaded dataset')} · ${escapeHtml(model.model_name)} v${escapeHtml(model.version||'1.0')} · Compatibility ${(Number(data.compatibility_score||0)*100).toFixed(0)}%</span></div></div>`;
  } else if(state.trainedOnUpload && state.trainingResult){
    const r=state.trainingResult;
    area.innerHTML=`<div class="banner">🎉 <div><strong>New model trained successfully</strong><span> · CV AUC ${escapeHtml(r.auc_cv)} · ${escapeHtml(r.n_features)} features · ${escapeHtml(r.duration_seconds)}s · Hash: ${escapeHtml(r.dataset_hash)}</span></div></div>`;
  } else if(data.reused_model) {
    area.innerHTML=`<div class="banner">⚡ <div><strong style="color:var(--blue)">Existing model reused</strong><span> · Same schema detected — skipped retraining.</span></div></div>`;
  } else {
    area.innerHTML=`<div class="banner">✓ <div><strong>Default model active</strong><span> · Exact schema match — no training required.</span></div></div>`;
  }
}

async function initDashboard() {
  state = await requireAnalytics();
  if (!state) return;
  data = state.payload; customers = data.customers || [];
  basePage('overview',`
    <div class="page-head"><div><h1>Overview</h1><p>Monitor customer churn risk and understand the factors driving retention.</p></div></div>
    <div id="banner"></div>
    <section class="card data-source-card"><div><span class="eyebrow">Current data source</span><h3>${escapeHtml(data.source_file||'Customer dataset')}</h3><p>${(data.summary?.total_customers||customers.length).toLocaleString()} customers &middot; ${data.dataset_type==='unlabeled'?'Prediction dataset':'Training dataset'}</p></div><div><span class="muted">Model</span><strong>${escapeHtml(data.model?.model_name||'Active churn model')}</strong></div><div><span class="muted">Compatibility</span><strong>${data.compatibility_score!=null?Math.round(data.compatibility_score*100)+'%':'Ready'}</strong></div></section>
    <div class="kpi-grid" id="kpis"></div>
    <div class="chart-grid">
      <section class="card chart-card"><div class="card-title">Churn Timeline</div><div class="donut-wrap"><svg id="timeline-donut" width="120" height="120"></svg><div class="donut-legend" id="timeline-legend"></div></div></section>
      <section class="card chart-card"><div class="card-title">Risk Level</div><div class="donut-wrap"><svg id="risk-donut" width="120" height="120"></svg><div class="donut-legend" id="risk-legend"></div></div></section>
      <section class="card chart-card"><div class="card-title">How likely is each customer to churn?<span class="card-note">customers grouped by churn chance (0–100%)</span></div><div class="histogram" id="histogram"></div></section>
      <section class="card wide-card"><div class="card-title">What drives churn the most?<span class="card-note">each factor's influence on the model's prediction</span></div><div class="bar-list" id="drivers"></div></section>
    </div>
    <section class="card high-risk-card"><div class="card-title">Customers to review now <button class="text-action" onclick="go('customers.html')">View all customers</button></div><div class="table-wrap"><table class="insight-table"><thead><tr><th>Customer</th><th>Churn probability</th><th>Risk</th><th></th></tr></thead><tbody id="high-risk-list"></tbody></table></div></section>
    <div class="card"><div class="card-title">Quick navigation</div><div class="grid-3">
      <button class="btn btn-secondary" onclick="go('customers.html')">View Customers →</button>
      <button class="btn btn-secondary" onclick="go('segments.html')">Explore Segments →</button>
      <button class="btn btn-secondary" onclick="go('actions.html')">See Retention Actions →</button>
    </div></div>
    <div class="modal-overlay" id="list-modal" onclick="if(event.target===this)hideModal('list-modal')">
      <div class="modal"><div class="modal-head"><h3 id="list-title">Customers</h3><button class="modal-close" aria-label="Close list" onclick="hideModal('list-modal')">×</button></div><div class="modal-body" id="list-body"></div></div>
    </div>
  `);
  renderBanner();renderKPIs();
  const t=timelineData(),r=riskData();
  renderDonut('timeline-donut',t,timelineColors);renderDonut('risk-donut',r,riskColors);
  renderLegend('timeline-legend',t,timelineColors,showSegmentList);
  renderLegend('risk-legend',r,riskColors,showSegmentList);
  renderHistogram();renderDrivers();renderHighRiskCustomers();
}
document.addEventListener('DOMContentLoaded',initDashboard);
