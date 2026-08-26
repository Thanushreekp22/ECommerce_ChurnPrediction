let state=null,data=null,customers=[],filter='all',page=1;
const PAGE_SIZE=20;
let appliedStrategies={}; // applied strategies mirrored from the backend (key: customerId::strategyKey)

function filteredCustomers(){
  const q=(document.getElementById('customer-search')?.value||'').toLowerCase();
  return customers.filter(c=>{
    const id=String(c.customer_id||'').toLowerCase();
    if(q&&!id.includes(q))return false;
    if(filter==='churn')return c.churn_probability>=.5;
    if(filter==='high')return c.risk_level==='High';
    if(filter==='medium')return c.risk_level==='Medium';
    if(filter==='low')return c.risk_level==='Low';
    return true;
  });
}
function setCustomerFilter(value,btn){
  filter=value;page=1;
  document.querySelectorAll('.filter-btn').forEach(x=>x.classList.remove('active'));
  btn.classList.add('active');renderTable();
}
function probColor(p){return p>=.7?'var(--red)':p>=.4?'var(--amber)':'var(--green)'}

function renderTable(){
  const schema=data.schema||{}, cols=(schema.feature_cols||[]).slice(0,6);
  const list=filteredCustomers(), pages=Math.max(1,Math.ceil(list.length/PAGE_SIZE));
  if(page>pages)page=1;
  const slice=list.slice((page-1)*PAGE_SIZE,page*PAGE_SIZE);
  document.getElementById('table-head').innerHTML=`<tr><th>Customer ID</th><th>Churn Probability</th><th>Risk</th><th>Timeline</th>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}<th></th></tr>`;
  document.getElementById('table-body').innerHTML=slice.map(c=>`
    <tr class="clickable-row" data-id="${escapeHtml(c.customer_id)}">
      <td class="mono" style="color:var(--accent)">${escapeHtml(c.customer_id)}</td>
      <td><div class="prob-bar"><div class="prob-mini"><div class="prob-mini-fill" style="width:${(c.churn_probability*100).toFixed(0)}%;background:${probColor(c.churn_probability)}"></div></div><span class="mono">${(c.churn_probability*100).toFixed(1)}%</span></div></td>
      <td><span class="risk-badge ${c.risk_level}">${escapeHtml(c.risk_level)}</span></td>
      <td class="muted">${escapeHtml(c.churn_timeline)}</td>
      ${cols.map(col=>`<td class="muted">${escapeHtml(c[col])}</td>`).join('')}
      <td class="row-actions"><button class="btn btn-secondary row-view" data-view="${escapeHtml(c.customer_id)}" title="Open retention strategies for ${escapeHtml(c.customer_id)}">View</button></td>
    </tr>`).join('') || `<tr><td colspan="${5+cols.length}" class="empty">No customers match the selected filter.</td></tr>`;
  document.querySelectorAll('.clickable-row').forEach(row=>row.onclick=()=>showDetail(row.dataset.id));
  document.querySelectorAll('[data-view]').forEach(btn=>btn.onclick=event=>{event.stopPropagation();showDetail(btn.dataset.view);});
  const pageButtons=[];
  for(let i=1;i<=pages;i++)if(i===1||i===pages||Math.abs(i-page)<=2)pageButtons.push(`<button class="page-btn ${i===page?'active':''}" onclick="page=${i};renderTable()">${i}</button>`);
  document.getElementById('pagination').innerHTML=`<span class="muted" style="font-size:12px">${list.length} customers</span>${pageButtons.join('')}`;
  const s=data.summary||{};
  document.getElementById('customer-summary').innerHTML=`
    <div class="summary-chip">All <strong>${s.total_customers||customers.length}</strong></div>
    <div class="summary-chip">Predicted churn <strong>${s.predicted_churn||0}</strong></div>
    <div class="summary-chip">High risk <strong>${s.high_risk||0}</strong></div>
    <div class="summary-chip">Low risk <strong>${s.low_risk||0}</strong></div>`;
}

function generateRecs(c){
  // Prefer backend-generated personalised recommendations when available.
  if (c.recommendations && c.recommendations.length) {
    return c.recommendations.slice(0, 3).map(r => ({
      priority: r.priority || 'Low',
      action: r.recommendation || r.action || 'Schedule a proactive customer-success check-in.',
      driven_by: r.driven_by || null,
      strategy_key: r.driven_by ? String(r.driven_by).toLowerCase().split('_')[0] : 'general'
    }));
  }
  const recs=[];
  (c.top_shap_features||[]).forEach(s=>{
    const f=String(s.feature).toLowerCase(),inc=s.direction==='increases';
    if(f.includes('tenure')&&inc)recs.push({priority:'High',action:'Enrol in an early-tenure loyalty programme and schedule a 30-day check-in.'});
    else if(f.includes('complain')&&inc)recs.push({priority:'High',action:'Escalate the complaint and consider a service-recovery offer.'});
    else if(f.includes('cashback')&&!inc)recs.push({priority:'High',action:'Offer enhanced cashback or rewards for the next few orders.'});
    else if(f.includes('order')&&inc)recs.push({priority:'Medium',action:'Send a personalised re-order nudge with a return-customer offer.'});
    else if(f.includes('satisf')&&inc)recs.push({priority:'Medium',action:'Collect satisfaction feedback and follow up on the identified issue.'});
    else if(f.includes('day')||f.includes('since'))recs.push({priority:'Medium',action:'Trigger a win-back campaign with a personalised offer.'});
    else if(f.includes('address')||f.includes('device'))recs.push({priority:'Low',action:'Encourage multi-channel engagement through app notifications.'});
    else recs.push({priority:'Medium',action:`Monitor "${s.feature}" because it is a major churn driver for this customer.`});
  });
  return (recs.length?recs:[{priority:'Medium',strategy_key:'general',action:'Schedule a proactive customer-success check-in.'},{priority:'Low',strategy_key:'general',action:'Enrol in a standard retention communication sequence.'}]).slice(0,3);
}

function recKey(customerId,rec){
  return `${customerId}::${rec.strategy_key||'general'}`;
}
function wireApplyButtons(){
  const body=document.getElementById('detail-body');if(!body)return;
  const buttons=[...body.querySelectorAll('[data-apply-key]')];
  const refreshMeta=()=>{
    const meta=document.getElementById('applied-meta');if(!meta)return;
    if(!buttons.length){meta.innerHTML='';return;}
    const done=buttons.filter(b=>b.classList.contains('applied')).length;
    meta.innerHTML=`<span class="mono">${done}/${buttons.length} applied</span> · <a href="javascript:void(0)" id="apply-all-link">apply all</a> · <a href="javascript:void(0)" id="clear-applied-link">reset</a>`;
    const all=document.getElementById('apply-all-link');
    if(all)all.onclick=async()=>{
      try{
        for(const b of buttons.filter(x=>!x.classList.contains('applied')))await applyToggle(b,true);
        toast('All strategies marked as applied ✓');refreshMeta();
      }catch(e){alert('Could not save: '+e.message);}
    };
    const clear=document.getElementById('clear-applied-link');
    if(clear)clear.onclick=async()=>{
      try{
        for(const b of buttons.filter(x=>x.classList.contains('applied')))await applyToggle(b,false);
        toast('Strategy states reset');refreshMeta();
      }catch(e){alert('Could not save: '+e.message);}
    };
  };
  const applyToggle=async(btn,applied)=>{
    const cid=btn.dataset.cid, skey=btn.dataset.skey;
    // Server-side persistence so the applied state reflects on every device/page.
    await applyStrategyOnServer(skey,[cid],btn.dataset.action||null,applied);
    const key=`${cid}::${skey}`;
    if(applied)appliedStrategies[key]={appliedAt:new Date().toISOString()};else delete appliedStrategies[key];
    btn.classList.toggle('applied',applied);
    btn.textContent=applied?'✓ Applied':'Apply';
    btn.setAttribute('aria-label',applied?'Mark strategy as not applied':'Mark strategy as applied');
  };
  buttons.forEach(btn=>{
    btn.onclick=async()=>{
      const nowApplied=!btn.classList.contains('applied');
      try{
        await applyToggle(btn,nowApplied);refreshMeta();
        const notifyRow=btn.closest('.rec-card')?.querySelector('.notify-row');if(notifyRow)notifyRow.style.display=nowApplied?'':'none';
        toast(nowApplied?'Retention strategy marked as applied ✓':'Strategy marked as not applied');
      }catch(e){alert('Could not save the strategy state.\n\n'+e.message);}
    };
  });
  // Outreach: email / WhatsApp the applied strategy to this customer
  body.querySelectorAll('[data-notify-key]').forEach(nb=>{
    nb.onclick=async()=>{
      const cid=nb.dataset.notifyCid, skey=nb.dataset.notifySkey,
            action=nb.dataset.notifyAction, channel=nb.dataset.notifyChannel;
      nb.disabled=true; const orig=nb.textContent; nb.textContent='Sending…';
      try{
        const res=await notifyStrategy(cid,skey,action,channel);
        const m=(res.messages||[])[0]||{};
        if(res.wa_link){
          toast('WhatsApp link ready — opening…');
          setTimeout(()=>window.open(res.wa_link,'_blank'),300);
        }else if(channel==='email'&&res.sent!==false){
          toast(res.mode==='demo' ? 'Email sent (demo mode) — logged to outbox' : 'Email sent to '+cid);
        }else{
          toast('Could not send: '+(m.error||'unknown error'));
        }
      }catch(e){alert('Notification failed: '+e.message);}
      nb.disabled=false; nb.textContent=orig;
    };
  });
  refreshMeta();
}

function showDetail(id){
  const c=customers.find(x=>String(x.customer_id)===String(id));if(!c)return;
  const cols=data.schema?.feature_cols||[],prob=(c.churn_probability*100).toFixed(1);
  const profile=cols.map(col=>`<div class="profile-item"><label>${escapeHtml(col)}</label><span>${escapeHtml(c[col])}</span></div>`).join('');
  const shap=c.top_shap_features||[],max=Math.max(...shap.map(x=>Math.abs(x.shap_value)),.001);
  const shapHtml=shap.map(s=>`<div class="shap-row"><div class="shap-label" title="${escapeHtml(s.feature)}">${escapeHtml(s.feature)}</div><div class="shap-bar-wrap"><div class="shap-bar-fill" style="width:${Math.abs(s.shap_value)/max*100}%;background:${s.direction==='increases'?'var(--red)':'var(--green)'}"></div></div><div class="shap-val">${s.shap_value>0?'+':''}${Number(s.shap_value).toFixed(3)}</div></div>`).join('');
  const recs=generateRecs(c);
  const recsHtml=recs.map(r=>{
    const key=recKey(c.customer_id,r),applied=!!appliedStrategies[key];
    return `<div class="rec-card">
      <div class="rec-priority ${r.priority}">${r.priority}</div>
      <div style="flex:1;min-width:0"><div>${escapeHtml(r.action)}</div>${r.driven_by?`<div class="mono rec-driver">driven by ${escapeHtml(r.driven_by)}</div>`:''}</div>
      <button class="apply-btn ${applied?'applied':''}" data-apply-key="${escapeHtml(key)}" data-cid="${escapeHtml(c.customer_id)}" data-skey="${escapeHtml(r.strategy_key||'general')}" data-action="${escapeHtml(r.action)}" aria-label="${applied?'Mark strategy as not applied':'Mark strategy as applied'}">${applied?'✓ Applied':'Apply'}</button>
      <div class="notify-row" ${applied?'':'style="display:none"'}>
        <button class="btn btn-mini" data-notify-key="${escapeHtml(key)}" data-notify-cid="${escapeHtml(c.customer_id)}" data-notify-skey="${escapeHtml(r.strategy_key||'general')}" data-notify-action="${escapeHtml(r.action)}" data-notify-channel="email" title="Email this retention strategy to the customer">&#9993; Email</button>
        <button class="btn btn-mini" data-notify-key="${escapeHtml(key)}" data-notify-cid="${escapeHtml(c.customer_id)}" data-notify-skey="${escapeHtml(r.strategy_key||'general')}" data-notify-action="${escapeHtml(r.action)}" data-notify-channel="whatsapp" title="Open WhatsApp with the strategy pre-filled">WhatsApp</button>
      </div>
    </div>`;
  }).join('');
  const appliedCount=recs.filter(r=>appliedStrategies[recKey(c.customer_id,r)]).length;
  document.getElementById('detail-title').textContent=`Customer ${c.customer_id}`;
  document.getElementById('detail-body').innerHTML=`
    <div class="banner" style="margin-bottom:18px">
      <div><strong>${prob}% churn probability</strong><span> · ${escapeHtml(c.risk_level)} risk · ${escapeHtml(c.churn_timeline)}</span></div>
    </div>
    <div class="section-title">Customer Profile</div><div class="profile-grid">${profile||'<div class="muted">No profile fields returned.</div>'}</div>
    <div class="section-title">Top Churn Drivers (SHAP)</div>${shapHtml||'<div class="muted">SHAP values not available.</div>'}
    <div class="section-title">Recommended Retention Strategies <span class="applied-meta" id="applied-meta"></span></div>
    ${recs.length?recsHtml:'<div class="muted">No retention strategies available for this customer yet.</div>'}`;
  wireApplyButtons();
  showModal('detail-modal');
}

async function initCustomers(){
  state = await requireAnalytics();
  if (!state) return;
  data = state.payload; customers = data.customers || [];
  // Load the server-side audit trail and mirror it into a quick-lookup map.
  appliedStrategies = await fetchAppliedStrategies().then(res => {
    const map = {};
    (res.records || []).forEach(r => { map[`${r.customer_id}::${r.strategy_key}`] = { appliedAt: r.applied_at, action: r.action }; });
    return map;
  }).catch(error => {
    console.warn('Could not load applied strategies from backend:', error);
    return {};
  });
  basePage('customers',`
    <div class="page-head"><div><span class="eyebrow">Customer intelligence</span><h1>Customers</h1><p>Search, filter and inspect individual customer churn predictions.</p></div><button class="btn btn-secondary" onclick="go('actions.html')">View retention actions</button></div>
    <div class="customer-summary" id="customer-summary"></div>
    <section class="card customer-card"><div class="inner">
      <div class="search-row">
        <input class="search-box" id="customer-search" placeholder="Search customer ID..." />
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="churn">Will Churn</button>
        <button class="filter-btn" data-filter="high">High Risk</button>
        <button class="filter-btn" data-filter="medium">Medium Risk</button>
        <button class="filter-btn" data-filter="low">Low Risk</button>
      </div>
    </div>
    <div class="table-wrap"><table class="data-table"><thead id="table-head"></thead><tbody id="table-body"></tbody></table></div>
    <div class="pagination" id="pagination"></div></section>
    <div class="modal-overlay" id="detail-modal" onclick="if(event.target===this)hideModal('detail-modal')">
      <div class="modal"><div class="modal-head"><h3 id="detail-title">Customer Detail</h3><button class="modal-close" aria-label="Close customer details" onclick="hideModal('detail-modal')">×</button></div><div class="modal-body" id="detail-body"></div></div>
    </div>
  `);
  document.getElementById('customer-search').oninput=()=>{page=1;renderTable()};
  document.querySelectorAll('.filter-btn').forEach(btn=>btn.onclick=()=>setCustomerFilter(btn.dataset.filter,btn));
  try {
    const focus = await loadCustomerFocus();
    if (focus) {
      await clearCustomerFocus();
      setTimeout(() => showDetail(focus), 50);
    }
  } catch (error) {
    console.error('Could not load customer focus:', error);
  }
  renderTable();
}
document.addEventListener('DOMContentLoaded',initCustomers);
