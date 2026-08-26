/*
frontend/js/actions.js
Retention-strategy planner.

Builds a dynamic catalogue of interventions from the scored customers:
  A) groups every per-customer SHAP recommendation by its root driver
  B) adds rule-based cohorts detected in raw fields (new tenure, complaints,
     low satisfaction, inactivity, low cashback, single-order, low app hours)
  C) guarantees a catch-all outreach play covering any at-risk customer that
     no specific strategy touches

Every strategy reports how many customers it targets and an ESTIMATED number
that a campaign could retain (SAVE_RATE assumption, clearly footnoted).
Clicking a card opens a drill-down listing the affected customers.
*/

const SAVE_RATE = 0.30; // assumed campaign success rate used for estimates
const PRIORITY_ORDER = { High: 0, Medium: 1, Low: 2 };

function num(v){ if (v == null) return null; const n = typeof v === 'number' ? v : parseFloat(v); return Number.isNaN(n) ? null : n; }
function baseOf(driven){ const s = String(driven || ''); const i = s.indexOf('_'); return (i > 0 ? s.slice(0, i) : s).toLowerCase(); }

const STRATEGY_LIBRARY = {
  complain:              { icon:'🛠️', title:'Complaint resolution fast-track',    desc:'Assign a dedicated support agent, respond within 24 hours and pair the fix with a goodwill credit on the next order.' },
  tenure:                { icon:'🎓', title:'Early-tenure onboarding programme',  desc:'Guide young accounts with onboarding check-ins, milestone rewards and proactive success calls in their first months.' },
  satisfactionscore:     { icon:'⭐', title:'Service-recovery & satisfaction lift', desc:'Survey dissatisfied customers within 48 hours and offer a tailored make-good on their next order.' },
  dayssincelastorder:    { icon:'⏰', title:'Win-back reactivation sequence',      desc:'Re-engage inactive buyers with personalised product picks and a time-limited comeback offer.' },
  recency:               { icon:'⏰', title:'Win-back reactivation sequence',      desc:'Re-engage inactive buyers with personalised product picks and a time-limited comeback offer.' },
  tenure_days:           { icon:'🎓', title:'Early-tenure onboarding programme',  desc:'Guide recently-acquired accounts with onboarding check-ins and a second-purchase reward.' },
  frequency:             { icon:'📦', title:'Purchase-frequency booster',         desc:'Use bundles or subscribe-and-save nudges to raise order cadence for single-order accounts.' },
  total:                 { icon:'💎', title:'Purchase-value retention plan',      desc:'Protect spend-driven accounts with VIP check-ins and an exclusive retention perk.' },
  monetary:              { icon:'💰', title:'High-value customer retention',      desc:'Shield your most valuable accounts with premium benefits, free shipping and loyalty perks.' },
  average:               { icon:'🏆', title:'Premium VIP experience programme',   desc:'Offer curated, concierge-style benefits to at-risk premium-order customers.' },
  cashbackamount:        { icon:'💰', title:'Rewards & cashback boost',           desc:'Highlight and top-up rewards so at-risk customers immediately feel more value from staying.' },
  couponused:            { icon:'🎟️', title:'Personalised coupon bundles',        desc:'Send curated coupon packs aligned to each customer’s favourite category to spark the next purchase.' },
  ordercount:            { icon:'📦', title:'Purchase-frequency booster',         desc:'Use bundles or subscribe-and-save nudges to raise order cadence for single-order accounts.' },
  hourspendonapp:        { icon:'📱', title:'In-app engagement push',             desc:'Trigger curated in-app deals and content to lift session time and product discovery.' },
  numberofdeviceregistered:{ icon:'🖥️', title:'Multi-device stickiness',         desc:'Offer a bonus for registering extra devices to deepen platform habit.' },
  warehousetohome:       { icon:'🚚', title:'Delivery-experience fix',            desc:'Review logistics pain for far-distance customers; add proactive delivery updates and compensation.' },
  monthlycharges:        { icon:'🧾', title:'Plan-value review',                  desc:'Review pricing fit and move at-risk accounts to better-value plans before cancellation.' },
  general:               { icon:'🤝', title:'Proactive outreach programme',       desc:'A personal retention call or tailored offer for high-value at-risk customers without a specific trigger.' },
};

/* Cohort rules scanned across at-risk customers so strategies adapt to the data. */
const COHORT_RULES = [
  { base:'complain',            prio:'High',   match:c => { const v = num(c.Complain ?? c.complain ?? c.complaints); return v != null && v >= 1; } },
  { base:'satisfactionscore',   prio:'High',   match:c => { const v = num(c.SatisfactionScore ?? c.satisfaction); return v != null && v <= 2; } },
  { base:'tenure',              prio:'High',   match:c => {
      const m = num(c.Tenure ?? c.tenure_months);           if (m != null && m < 6) return true;
      const d = num(c.tenure_days ?? c.account_age_days);   return d != null && d < 180; } },
  { base:'dayssincelastorder',  prio:'Medium', match:c => { const v = num(c.DaySinceLastOrder ?? c.last_activity_days); return v != null && v >= 15; } },
  { base:'cashbackamount',      prio:'Medium', match:c => { const v = num(c.CashbackAmount ?? c.cashback_amount); return v != null && v < 100; } },
  { base:'couponused',          prio:'Medium', match:c => { const v = num(c.CouponUsed ?? c.coupon_used); return v != null && v === 0; } },
  { base:'ordercount',          prio:'Medium', match:c => { const v = num(c.OrderCount ?? c.number_of_orders); return v != null && v <= 1; } },
  { base:'hourspendonapp',      prio:'Medium', match:c => { const v = num(c.HourSpendOnApp ?? c.hour_spend_on_app); return v != null && v > 0 && v <= 1; } },
  // RFM / transaction cohorts (Shopify, WooCommerce, mock store, CSV order imports).
  // Keys match baseOf() on the recommender's driven_by values so cards merge.
  { base:'recency',             prio:'High',   match:c => { const v = num(c.recency); return v != null && v >= 30; } },
  { base:'frequency',           prio:'Medium', match:c => { const v = num(c.frequency ?? c.total_orders); return v != null && v <= 1; } },
  { base:'total',               prio:'Medium', match:c => { const v = num(c.total_spend ?? c.monetary); return v != null && v > 200; } },
  { base:'average',             prio:'Low',    match:c => { const v = num(c.average_order_value); return v != null && v > 50; } },
];

function ensureStrategy(cat, base){
  if (!cat[base]) {
    const lib = STRATEGY_LIBRARY[base] || {};
    cat[base] = {
      key: base,
      icon: lib.icon || '🎯',
      title: lib.title || ('Targeted retention play: ' + base.replace(/_/g,' ')),
      desc: lib.desc || '',
      priority: 'Low',
      customers: new Set(),
      probs: [],
      examples: []
    };
  }
  return cat[base];
}

function buildStrategies(payload){
  const customers = payload.customers || [];
  const cat = {};
  const isAtRisk = c => (c.churn_probability || 0) >= 0.5 || c.risk_level === 'High';
  const atRisk = customers.filter(isAtRisk);

  // A) SHAP-driven recommendations grouped by root driver
  customers.forEach(c => {
    (c.recommendations || []).forEach(r => {
      const base = baseOf(r.driven_by) || 'general';
      const e = ensureStrategy(cat, base);
      e.customers.add(c.customer_id);
      e.probs.push(c.churn_probability || 0);
      if (!e.desc && r.recommendation) e.desc = r.recommendation;
      const p = r.priority || 'Medium';
      if ((PRIORITY_ORDER[p] ?? 1) < (PRIORITY_ORDER[e.priority] ?? 1)) e.priority = p;
      if (e.examples.length < 100) e.examples.push({ customer_id:c.customer_id, prob:c.churn_probability||0, risk:c.risk_level||'Medium' });
    });
  });

  // B) Rule-based cohorts from raw fields (adapt to each dataset)
  atRisk.forEach(c => {
    COHORT_RULES.forEach(rule => {
      if (!rule.match(c)) return;
      const e = ensureStrategy(cat, rule.base);
      e.customers.add(c.customer_id);
      e.probs.push(c.churn_probability || 0);
      if ((PRIORITY_ORDER[rule.prio] ?? 1) < (PRIORITY_ORDER[e.priority] ?? 1)) e.priority = rule.prio;
      if (e.examples.length < 100) e.examples.push({ customer_id:c.customer_id, prob:c.churn_probability||0, risk:c.risk_level||'Medium' });
    });
  });

  // C) Catch-all: every remaining at-risk customer gets an actionable play
  const covered = new Set();
  Object.entries(cat).forEach(([base, e]) => { if (base !== 'general') e.customers.forEach(id => covered.add(id)); });
  const uncovered = atRisk.filter(c => !covered.has(c.customer_id));
  if (uncovered.length) {
    const e = ensureStrategy(cat, 'general');
    uncovered.forEach(c => {
      e.customers.add(c.customer_id);
      e.probs.push(c.churn_probability || 0);
      if (e.examples.length < 100) e.examples.push({ customer_id:c.customer_id, prob:c.churn_probability||0, risk:c.risk_level||'Medium' });
    });
  }

  const strategies = Object.values(cat)
    .map(e => {
      const affected = e.customers.size;
      const avgProb = e.probs.length ? e.probs.reduce((a,b)=>a+b,0)/e.probs.length : 0;
      return {
        key:e.key, icon:e.icon, title:e.title, desc:e.desc || STRATEGY_LIBRARY[e.key]?.desc || '',
        priority:e.priority, affected, estSaved: Math.round(affected * SAVE_RATE),
        avgProbability: Math.round(avgProb*100), customerIds:[...e.customers], examples:e.examples
      };
    })
    .filter(s => s.affected > 0)
    .sort((a,b) => (PRIORITY_ORDER[a.priority]-PRIORITY_ORDER[b.priority]) || (b.affected-a.affected));

  const totalAtRisk = payload.summary?.predicted_churn != null
    ? payload.summary.predicted_churn : atRisk.length;
  const coveredIds = new Set(); strategies.forEach(s => s.customerIds.forEach(id => coveredIds.add(id)));
  const stats = {
    totalAtRisk,
    covered: [...coveredIds].filter(id => atRisk.some(c => String(c.customer_id)===String(id))).length,
    estSavedTotal: strategies.reduce((a,s)=>a+s.estSaved,0),
    count: strategies.length
  };
  return { strategies, stats };
}

/* ── Rendering ───────────────────────────────────────────────────────────── */
let sortBy='impact', prioFilter='all', showAll=false, serverApplied={}, catalog={strategies:[],stats:{}};
let customerById={};

async function setStrategyForAll(s, applied){

  await applyStrategyOnServer(s.key, s.customerIds, s.desc || s.title, applied);

  s.customerIds.forEach(id => {

    const k = id + '::' + s.key;

    if (applied) serverApplied[k] = { appliedAt: new Date().toISOString(), action: s.title };

    else delete serverApplied[k];

  });

}

function appliedCountFor(s){ return s.customerIds.filter(id => serverApplied[id + '::' + s.key]).length; }

function strategyCardHtml(s){
  const appliedN = appliedCountFor(s);
  return `<section class="card strat-card prio-${s.priority}" data-key="${escapeHtml(s.key)}">
    <div class="strat-head">
      <span class="strat-icon" aria-hidden="true">${s.icon}</span>
      <div style="flex:1;min-width:0">
        <span class="prio-badge ${s.priority}">${s.priority} priority</span>
        <h3>${escapeHtml(s.title)}</h3>
      </div>
      ${appliedN>0?`<span class="applied-chip">✓ ${appliedN}/${s.affected} applied</span>`:`<button class="apply-btn" data-apply-strategy="${escapeHtml(s.key)}" aria-label="Apply this strategy to all targeted customers">Apply to ${s.affected}</button>`}
    </div>
    <p class="strat-desc">${escapeHtml(s.desc)}</p>
    <div class="strat-stats">
      <div><strong>${s.affected.toLocaleString()}</strong><span>customers targeted</span></div>
      <div><strong>≈${s.estSaved.toLocaleString()}</strong><span>est. could be saved*</span></div>
      <div><strong>${s.avgProbability}%</strong><span>avg churn chance</span></div>
    </div>
    <button class="text-action" data-open-strategy="${escapeHtml(s.key)}">View affected customers →</button>
  </section>`;
}

function overviewHtml(stats){
  const appliedCustomers = new Set(Object.keys(serverApplied).map(k => k.split('::')[0])).size;
  return `
  <div class="kpi-grid">
    <div class="kpi-card kpi-danger"><div class="kpi-label">At-risk customers</div><div class="kpi-value">${stats.totalAtRisk.toLocaleString()}</div><div class="kpi-sub">predicted to churn</div></div>
    <div class="kpi-card kpi-info"><div class="kpi-label">Actionable strategies</div><div class="kpi-value">${stats.count}</div><div class="kpi-sub">adapted to this dataset</div></div>
    <div class="kpi-card kpi-warn"><div class="kpi-label">Est. savable*</div><div class="kpi-value">≈${stats.estSavedTotal.toLocaleString()}</div><div class="kpi-sub">across all campaigns</div></div>
    <div class="kpi-card kpi-good"><div class="kpi-label">Actions applied</div><div class="kpi-value">${appliedCustomers.toLocaleString()}</div><div class="kpi-sub">of ${stats.totalAtRisk.toLocaleString()} at-risk customers</div></div>
  </div>`;
}

function visibleStrategies(){
  let list=catalog.strategies.filter(s=>prioFilter==='all'||s.priority===prioFilter);
  if(sortBy==='customers')list=list.slice().sort((a,b)=>b.affected-a.affected);
  else if(sortBy==='saved')list=list.slice().sort((a,b)=>b.estSaved-a.estSaved);
  return list;
}

function renderStrategies(){
  document.getElementById('actions-overview').innerHTML=overviewHtml(catalog.stats);
  const list=visibleStrategies();
  const shown=showAll?list:list.slice(0,6);
  const el=document.getElementById('strategy-list');
  if(!list.length){el.innerHTML='<div class="card empty" style="padding:30px;text-align:center">🎉 No at-risk customers in this analysis — nothing to action right now.</div>';return;}
  el.innerHTML=shown.map(strategyCardHtml).join('')+
    (list.length>6?`<div style="text-align:center;margin-top:6px"><button class="btn btn-secondary" id="toggle-all-strategies">${showAll?'Show fewer':'Show all '+list.length+' strategies'}</button></div>`:'')+
    '<p class="est-note">* Savings assume a '+Math.round(SAVE_RATE*100)+'% campaign success rate on targeted customers; customers may appear in more than one strategy.</p>';
  document.getElementById('toggle-all-strategies')?.addEventListener('click',()=>{showAll=!showAll;renderStrategies();});
}

function openStrategy(key){
  const s=catalog.strategies.find(x=>x.key===key);if(!s)return;
  const rows=s.examples.map(x=>{
    const c=customerById[String(x.customer_id)]||{};
    return `<tr><td class="mono">${escapeHtml(x.customer_id)}</td><td><b>${((x.prob||0)*100).toFixed(0)}%</b></td><td><span class="risk-badge ${escapeHtml(c.risk_level||x.risk||'Medium')}">${escapeHtml(c.risk_level||x.risk||'Medium')}</span></td><td>${appliedFor(String(x.customer_id)) ? '<span class="applied-chip">yes</span>' : '-'}</td></tr>`;
  }).join('');
  const more=s.customerIds.length>s.examples.length?`<p class="muted" style="margin-top:10px;font-size:12px">+ ${(s.customerIds.length-s.examples.length)} more customers in this group.</p>`:'';
  const appliedFor = id => !!serverApplied[id + '::' + s.key];
  const allApplied = s.customerIds.length > 0 && s.customerIds.every(appliedFor);
  const appliedN = appliedCountFor(s);
  document.getElementById('strategy-title').textContent=s.title;
  document.getElementById('strategy-body').innerHTML=`
    <div class="muted" id="strategy-outbox-note" style="display:none;margin-bottom:10px;font-size:12px"></div>
    <div class="banner" style="margin-bottom:16px"><div><strong>${s.icon} ${s.affected.toLocaleString()} customers targeted</strong><span> · est. ≈${s.estSaved.toLocaleString()} could be saved* · avg churn chance ${s.avgProbability}%</span></div></div>
    <p style="margin-bottom:14px;color:var(--muted);line-height:1.6">${escapeHtml(s.desc)}</p>
    <div class="table-wrap"><table class="insight-table"><thead><tr><th>Customer</th><th>Churn probability</th><th>Risk</th><th>Applied</th></tr></thead><tbody>${rows}</tbody></table></div>${more}
    <div style="display:flex;gap:10px;margin-top:18px">
      <button class="apply-btn ${applied?'applied':''}" id="strategy-apply-btn" data-k="${escapeHtml(s.key)}">${applied?'✓ Applied':'Mark strategy as applied'}</button>
      <button class="btn btn-secondary" id="strategy-email-btn">&#9993; Email targeted customers</button>
      <button class="btn btn-secondary" id="strategy-wa-btn">WhatsApp (link)</button>
      <button class="btn btn-secondary" onclick="hideStrategy()">Close</button>
    </div>
    <p class="est-note">* Assumes a ${Math.round(SAVE_RATE*100)}% campaign success rate.</p>`;
  document.getElementById('strategy-apply-btn').onclick=function(){
    const nowApplied=!allApplied;
    setStrategyForAll(s,nowApplied).then(()=>{
            openStrategy(s.key);
      renderStrategies();toast(nowApplied?"Strategy marked as applied ✓":"Strategy unmarked");
    }).catch(e=>alert('Could not save: '+e.message));
  };
  document.getElementById('strategy-email-btn').onclick=async function(){
    const btn=this; btn.disabled=true; const orig=btn.textContent; btn.textContent='Sending…';
    const ids=s.customerIds.slice(0,15); let sent=0, demo=false, failed=0;
    try{
      for(const id of ids){
        try{
          const res=await notifyStrategy(String(id), s.key, s.title||s.desc, 'email');
          const m=(res.messages||[])[0]||{};
          if(res.sent!==false){ sent++; if(m.mode==='demo') demo=true; }
          else failed++;
        }catch(e){ failed++; }
      }
      toast(demo ? sent+' emails logged to outbox (demo mode)' : sent+' emails sent to targeted customers' + (failed?' · '+failed+' failed':''));
      document.getElementById('strategy-outbox-note').style.display=sent?'':'none';
    } catch(e){ alert('Notification failed: '+e.message); }
    btn.disabled=false; btn.textContent=orig;
  };
  document.getElementById('strategy-wa-btn').onclick=async function(){
    const id=String(s.customerIds[0]||''); if(!id){ toast('No customer for WhatsApp link.'); return; }
    const res=await notifyStrategy(id, s.key, s.title||s.desc, 'whatsapp');
    if(res.wa_link){ toast('Opening WhatsApp for '+id+' …'); setTimeout(()=>window.open(res.wa_link,'_blank'),300); }
    else { const m=(res.messages||[])[0]||{}; alert(m.error||'No phone number available. Add a phone column to enable WhatsApp.'); }
  };
  // Show the outbox note (how many demo-emails are logged)
  getNotificationOutbox().then(ob=>{
    const note=document.getElementById('strategy-outbox-note');
    if(note&&ob.messages&&ob.messages.length){ note.style.display=''; note.textContent=ob.messages.length+' outreach message(s) logged (demo mode). See them on the E-commerce page.'; }
  }).catch(()=>{});
  showModal('strategy-modal');
}
function hideStrategy(){hideModal('strategy-modal');}

async function initActions(){
  const state=await requireAnalytics();if(!state)return;
  await fetchAppliedStrategies().then(res=>{serverApplied={};(res.records||[]).forEach(r=>{serverApplied[r.customer_id+'::'+r.strategy_key]={appliedAt:r.applied_at,action:r.action};});}).catch(()=>{});
  catalog=buildStrategies(state.payload);
  customerById={};(state.payload.customers||[]).forEach(c=>{customerById[String(c.customer_id)]=c;});
  basePage('actions',`
    <div class="page-head"><div><h1>Retention Strategies</h1><p>Interventions adapted to this dataset's churn drivers — with the customers each one saves.</p></div></div>
    <div id="actions-overview"></div>
    <div class="strategy-toolbar">
      <span class="muted" style="font-size:12px">Priority:</span>
      <button class="filter-btn active" data-prio="all">All</button>
      <button class="filter-btn" data-prio="High">High</button>
      <button class="filter-btn" data-prio="Medium">Medium</button>
      <button class="filter-btn" data-prio="Low">Low</button>
      <span class="muted" style="font-size:12px;margin-left:auto">Sort:</span>
      <select id="strategy-sort" class="sort-select"><option value="impact">Biggest impact</option><option value="customers">Most customers</option><option value="saved">Most savable</option></select>
    </div>
    <div id="strategy-list" class="strat-grid"></div>
    <div class="modal-overlay" id="strategy-modal" onclick="if(event.target===this)hideStrategy()">
      <div class="modal"><div class="modal-head"><h3 id="strategy-title">Strategy</h3><button class="modal-close" aria-label="Close strategy details" onclick="hideStrategy()">×</button></div><div class="modal-body" id="strategy-body"></div></div>
    </div>`);
  document.querySelectorAll('.strategy-toolbar .filter-btn').forEach(b=>b.onclick=()=>{
    prioFilter=b.dataset.prio;showAll=false;
    document.querySelectorAll('.strategy-toolbar .filter-btn').forEach(x=>x.classList.toggle('active',x===b));
    renderStrategies();
  });
  document.getElementById('strategy-sort').onchange=e=>{sortBy=e.target.value;renderStrategies();};
  document.getElementById('strategy-list').addEventListener('click',e=>{
    const open=e.target.closest('[data-open-strategy]');if(open){openStrategy(open.dataset.openStrategy);return;}
    const ap=e.target.closest('[data-apply-strategy]');if(!ap)return;
    e.stopPropagation();
    const s=catalog.strategies.find(x=>x.key===ap.dataset.applyStrategy);if(!s)return;const nowApplied=true;
    setStrategyForAll(s,nowApplied).then(()=>{
      
      toast(nowApplied?'Strategy marked as applied ✓':'Strategy unmarked');renderStrategies();
    }).catch(err=>alert('Could not save: '+err.message));
  });
  renderStrategies();
}
document.addEventListener('DOMContentLoaded',initActions);
