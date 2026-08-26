function escapeHtml(value) {
  return String(value ?? '—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#039;");
}
function toast(message, tone='success') {
  let el=document.getElementById('toast');
  if(!el){el=document.createElement('div');el.id='toast';el.className='toast';el.setAttribute('role','status');el.setAttribute('aria-live','polite');document.body.appendChild(el);}
  el.textContent=message;el.dataset.tone=tone;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),3200);
}
/* Skeleton loading placeholders */
function skeletonKPIs(count){
  const n=count||4;let html='';
  for(let i=0;i<n;i++)html+='<div class="kpi-card skeleton skeleton-kpi" aria-hidden="true"></div>';
  return html;
}
function skeletonBoxes(count,height){
  const n=count||2,h=height||96;let html='';
  for(let i=0;i<n;i++)html+='<div class="card skeleton" style="min-height:'+h+'px" aria-hidden="true"></div>';
  return html;
}
function go(page){window.location.href=page;}
async function startNewDataset(){
  try{
    await clearAnalytics();
    await clearCustomerFocus();
  }catch(error){
    console.warn('Could not clear the previous browser analysis:', error);
  }
  go('upload.html');
}
async function requireAnalytics(){
  document.body.innerHTML='<main class="page-loading"><span><i></i>Loading customer analytics...</span></main>';
  try{const state=await loadAnalytics();if(state)return state;renderNoAnalysis();return null;}
  catch(error){renderAppError(error.message);return null;}
}
function showModal(id){document.getElementById(id)?.classList.add('active');}
function hideModal(id){document.getElementById(id)?.classList.remove('active');}
function renderNoAnalysis(){document.body.innerHTML='<main class="standalone-state"><div class="state-icon">↗</div><h1>No analysis available</h1><p>Upload a customer dataset to begin exploring churn risk — or run the built-in demo to see the full pipeline in action.</p><div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center"><button class="btn btn-primary" onclick="go(\'upload.html\')">Upload dataset</button><button class="btn btn-secondary" onclick="runInstantDemo()">⚡ Run instant demo</button></div></main>';}
async function runInstantDemo(){
  try{
    document.body.innerHTML='<main class="page-loading"><span><i></i>Running instant demo — pulling mock store orders &amp; scoring customers…</span></main>';
    await syncPlatform('mock');
    const payload=await predictPlatform('mock');
    await saveAnalytics(payload,false,null);
    go('index.html');
  }catch(error){
    renderAppError(error.message||'The instant demo could not run. Is the backend running on port 8000?');
  }
}
function renderAppError(message){document.body.innerHTML='<main class="standalone-state"><div class="state-icon error">!</div><h1>Unable to load analytics</h1><p>'+escapeHtml(message||'The stored analysis could not be opened.')+'</p><button class="btn btn-secondary" onclick="location.reload()">Try again</button></main>';}
function navItem(active,key,href,label,icon){return '<a class="nav-link '+(active===key?'active':'')+'" href="'+href+'">'+icon+'<span>'+label+'</span></a>';}
function buildNav(active){
  const el=document.getElementById('sidebar');if(!el)return;
  const icons={
    dashboard:'<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="9" rx="2"/><rect x="14" y="3" width="7" height="5" rx="2"/><rect x="14" y="12" width="7" height="9" rx="2"/><rect x="3" y="16" width="7" height="5" rx="2"/></svg>',
    customers:'<svg viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/></svg>',
    segments:'<svg viewBox="0 0 24 24"><path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6" rx="1"/><rect x="12.5" y="8" width="3" height="10" rx="1"/><rect x="18" y="5" width="3" height="13" rx="1"/></svg>',
    actions:'<svg viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>',
    models:'<svg viewBox="0 0 24 24"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="7"/></svg>',
    store:'<svg viewBox="0 0 24 24"><path d="M3 3h2l2.2 11.1a2 2 0 0 0 2 1.6h8.8a2 2 0 0 0 2-1.6L21 7H6"/><circle cx="10" cy="20" r="1"/><circle cx="18" cy="20" r="1"/></svg>'
  };
  el.innerHTML='<button class="nav-new" aria-label="Upload a new dataset" title="New dataset" onclick="startNewDataset()"><b>+</b><span>New dataset</span></button><div class="nav-links">'+navItem(active,'overview','index.html','Dashboard',icons.dashboard)+navItem(active,'customers','customers.html','Customers',icons.customers)+navItem(active,'segments','segments.html','Segments',icons.segments)+navItem(active,'actions','actions.html','Actions',icons.actions)+navItem(active,'model','model.html','Models',icons.models)+navItem(active,'ecommerce','ecommerce.html','E-commerce',icons.store)+'</div><div class="nav-bottom-dot" title="System ready"></div>';
}
function setupHeader(){
  getHealth().then(data=>{const name=document.getElementById('model-name-badge'),auc=document.getElementById('auc-badge');if(name)name.textContent=data.active_model||'Model unavailable';if(auc)auc.textContent=data.auc_cv?'AUC '+data.auc_cv:'No score';}).catch(()=>{const name=document.getElementById('model-name-badge');if(name)name.textContent='API offline';});
}
function basePage(active,content){
  document.body.innerHTML='<div class="app"><header class="topbar"><a class="logo" href="index.html">Churn<span>IQ</span></a><div class="topbar-right"><div class="source-badge"><i></i><span>Analysis ready</span></div><div class="model-badge"><span id="model-name-badge">Loading model</span><span id="auc-badge">—</span></div><button class="btn btn-primary" onclick="go(\'upload.html\')">Upload dataset</button></div></header><div class="shell"><aside class="sidebar" id="sidebar"></aside><main class="main">'+content+'</main></div></div>';
  buildNav(active);setupHeader();
}


/* Inline SVG favicon (keeps console clean, no extra file needed) */
(function(){
  const l=document.createElement('link');
  l.rel='icon';
  const svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    +'<rect width="100" height="100" rx="24" fill="#0a8f78"/>'
    +'<text x="50" y="70" font-size="54" text-anchor="middle" fill="white" '
    +'font-family="Arial,sans-serif" font-weight="bold">C</text></svg>';
  l.href='data:image/svg+xml,'+encodeURIComponent(svg);
  document.head.appendChild(l);
})();
