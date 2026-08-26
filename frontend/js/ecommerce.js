let transactionFile = null;

function formatNumber(value) { return Number(value || 0).toLocaleString(); }

async function refreshIntegration() {
  document.getElementById('integration-summary').innerHTML = skeletonKPIs(4);
  const summary = await getEcommerceSummary();
  document.getElementById('integration-summary').innerHTML = `
    <div class="card"><h3>Customers</h3><div class="kpi-value">${formatNumber(summary.customers)}</div></div>
    <div class="card"><h3>Orders</h3><div class="kpi-value">${formatNumber(summary.orders)}</div></div>
    <div class="card"><h3>Revenue</h3><div class="kpi-value">${formatNumber(summary.revenue)}</div></div>
    <div class="card"><h3>Last transaction</h3><p>${escapeHtml(summary.last_transaction || '—')}</p></div>`;
  document.getElementById('import-history').innerHTML = (summary.history || []).map(run =>
    `<div class="rec-card"><strong>${escapeHtml(run.source_name)}</strong><span>${escapeHtml(run.status)} · ${formatNumber(run.rows_inserted)} imported · ${escapeHtml(run.completed_at || '')}</span></div>`
  ).join('') || '<div class="empty">No imports yet.</div>';
}

async function refreshPlatformStatus() {
  try {
    const status = await getPlatformStatus();
    const platforms = status.platforms || {};
    const lastSync = status.last_sync || {};
    const row = (key, label, configured) => {
      const at = lastSync['platform:' + key];
      return `<div class="platform-item"><strong>${escapeHtml(label)}</strong>` +
        `<span class="mono" style="color:${configured ? 'var(--green)' : 'var(--amber)'}">${configured ? 'Configured' : 'Add env keys'}</span>` +
        (at ? `<span class="mono" style="font-size:10.5px;color:var(--muted)">synced ${escapeHtml(String(at).slice(0, 16).replace('T', ' '))}</span>` : '');
    };
    document.getElementById('platform-status').innerHTML =
      `<div class="platform-row">` +
      row('shopify', 'Shopify', !!(platforms.shopify && platforms.shopify.configured)) +
      row('woocommerce', 'WooCommerce', !!(platforms.woocommerce && platforms.woocommerce.configured)) +
      row('mock', 'Demo Mock Store', true) +
      `</div>`;
  } catch (error) {
    console.warn('Platform status unavailable:', error);
  }
}

async function doSync(name, label) {
  try {
    toast(`Syncing ${label}…`);
    const result = await syncPlatform(name);
    if (result.status === 'error') { alert(`${label} sync failed: ${result.error}`); return; }
    toast(`${label}: ${result.orders_imported || 0} new · ${result.orders_updated || 0} updated`);
    await refreshIntegration();
    await refreshPlatformStatus();
  } catch (error) {
    alert(error.message);
  }
}
async function syncMockStore(){ await doSync('mock', 'Demo store'); }
async function syncShopify(){ await doSync('shopify', 'Shopify'); }
async function syncWoo(){ await doSync('woocommerce', 'WooCommerce'); }

async function predictMockCustomers() {
  try {
    const response = await predictPlatform('mock');
    await saveAnalytics(response, false, null);
    go('index.html');
  } catch (error) {
    alert(error.message);
  }
}

async function analyzeTransactions() {
  if (!transactionFile) return;
  const result = await analyzeEcommerceFile(transactionFile);
  const mapped = Object.entries(result.mapping || {}).map(([field, value]) =>
    `<div class="rec-card"><strong>${escapeHtml(value.source)}</strong><span>→ ${escapeHtml(field)} · ${(value.confidence * 100).toFixed(0)}%</span></div>`
  ).join('');
  document.getElementById('mapping-preview').innerHTML = `${mapped}<p class="muted">${escapeHtml((result.warnings || []).join(' ') || 'Ready to import.')}</p>`;
  document.getElementById('import-button').disabled = !result.ready_to_import;
}

async function importTransactions() {
  const result = await importEcommerceFile(transactionFile);
  toast(`${result.orders_imported} orders imported.`);
  await refreshIntegration();
}

async function predictTransactions() {
  const response = await predictEcommerceCustomers();
  await saveAnalytics(response, false, null);
  go('index.html');
}

async function initEcommerce() {
  const card=`<section class="card"><div class="inner"><h3>Connect e-commerce platform</h3><p>Pull orders from a connected store (Shopify / WooCommerce) or use the built-in demo mock store. Purchases flow into customer features for churn scoring.</p>
    <div class="platform-row" id="platform-status">Loading...</div>
    <div class="inner-actions">
      <button class="btn btn-secondary" id="shopify-sync-button">🛍️ Sync Shopify</button>
      <button class="btn btn-secondary" id="woo-sync-button">🛒 Sync WooCommerce</button>
      <button class="btn btn-secondary" id="mock-sync-button">🧪 Demo store</button>
      <button class="btn btn-primary" id="mock-predict-button">Predict imported customers</button>
    </div>
  </div></section>`;
  basePage('ecommerce', `<div class="page-head"><div><h1>E-commerce Integration</h1><p>Preview, normalize, and import transaction data. Only compatible churn models can score the generated customer features.</p></div></div>
    ${card}
    <section class="card"><div class="inner"><h3>Import transaction CSV / Excel</h3><input id="transaction-file" type="file" accept=".csv,.xlsx,.xls"><button class="btn btn-secondary" id="analyze-button">Analyze</button><button class="btn btn-primary" id="import-button" disabled>Import data</button></div><div class="inner" id="mapping-preview"></div></section>
    <section class="grid-2" id="integration-summary"></section><section class="card"><div class="inner"><h3>Customer features</h3><p>Recency, frequency, monetary value, average order value, and tenure are built from imported orders without fabricated fields.</p><button class="btn btn-primary" id="predict-button">Predict churn</button></div></section><section class="card"><div class="inner"><h3>Retention outreach log</h3><p>Emails &amp; WhatsApp links generated when you apply retention strategies (demo mode writes to an outbox unless SMTP is configured).</p><div id="outreach-list"><div class="muted">No outreach yet - apply a strategy and use Email / WhatsApp to reach customers.</div></div><button class="btn btn-secondary" id="outreach-outbox-btn">Refresh outreach log</button></div></section><section class="card"><div class="inner"><h3>Import history</h3><div id="import-history"></div></div></section>`);
  document.getElementById('transaction-file').onchange = event => { transactionFile = event.target.files[0] || null; };
  document.getElementById('analyze-button').onclick = () => analyzeTransactions().catch(error => alert(error.message));
  document.getElementById('import-button').onclick = () => importTransactions().catch(error => alert(error.message));
  document.getElementById('predict-button').onclick = () => predictTransactions().catch(error => alert(error.message));
  document.getElementById('shopify-sync-button').onclick = () => syncShopify();
  document.getElementById('woo-sync-button').onclick = () => syncWoo();
  document.getElementById('mock-sync-button').onclick = () => syncMockStore();
  document.getElementById('mock-predict-button').onclick = () => predictMockCustomers();
  await refreshIntegration();
  await refreshPlatformStatus();
  await refreshOutreach();
  bindOutreachRefresh();
}
async function refreshOutreach(){
  try{
    const out = await getNotificationOutbox();
    const list = document.getElementById('outreach-list'); if(!list) return;
    const msgs = out.messages || [];
    list.innerHTML = msgs.length ? msgs.slice(0,20).map(m=>{
      const icon = m.channel==='whatsapp' ? String.fromCodePoint(0x1F4AC) : String.fromCodePoint(0x2709,0xFE0F);
      const type = m.channel==='whatsapp' ? '<span class="risk-badge High">WhatsApp link</span>' : '<span class="risk-badge Medium">Email</span>';
      return '<div class="rec-card"><div>'+icon+' '+type+'</div><div><strong>'+escapeHtml(m.recipient||'')+'</strong></div><div class="muted" style="font-size:12px">'+escapeHtml(m.subject||'')+'</div><div class="muted" style="font-size:11px">'+escapeHtml((m.body||'').slice(0,120))+((m.body||'').length>120?'\u2026':'')+'</div><div class="muted" style="font-size:11px">'+escapeHtml(m.note||'')+' \u00b7 '+escapeHtml(String(m.sent_at||'').slice(0,19).replace('T',' '))+'</div></div>';
    }).join('') : '<div class="muted">No outreach yet - apply a strategy and use Email / WhatsApp to reach customers.</div>';
  }catch(error){ console.warn('Outreach log unavailable:', error); }
}
function bindOutreachRefresh(){ const b=document.getElementById('outreach-outbox-btn'); if(b) b.onclick=()=>refreshOutreach().catch(e=>alert(e.message)); }
document.addEventListener('DOMContentLoaded', () => initEcommerce().catch(error => alert(error.message)));
