let uploadedFile=null;
let schemaAnalysis=null;

function setupDrop(){
  const zone=document.getElementById('drop-zone'),input=document.getElementById('file-input');
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('dragover')});
  zone.addEventListener('dragleave',()=>zone.classList.remove('dragover'));
  zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('dragover');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0])});
  input.addEventListener('change',e=>{if(e.target.files[0])handleFile(e.target.files[0])});
}
async function handleFile(file){
  const valid=['.csv','.xlsx','.xls'];
  if(!valid.some(x=>file.name.toLowerCase().endsWith(x))){alert('Please upload a CSV or Excel file.');return}
  try {
    await clearAnalytics();
    await clearCustomerFocus();
  } catch (error) {
    console.warn('Could not clear the previous analytics state:', error);
  }
  uploadedFile=file;
  document.getElementById('schema-page').style.display='block';
  document.getElementById('upload-section').style.display='none';
  document.getElementById('schema-subtitle').textContent=`"${file.name}" · ${(file.size/1024).toFixed(1)} KB · Ready to analyse`;
  document.getElementById('schema-grid').innerHTML=`
    <div class="schema-box" style="grid-column:1/-1">
      <h4>File detected</h4>
      <strong>${escapeHtml(file.name)}</strong>
      <p class="muted" style="margin-top:7px">ChurnIQ will automatically detect column types, identify the churn target and analyse the dataset.</p>
    </div>`;
  document.getElementById('schema-warning').innerHTML=`ℹ If this schema matches an existing model, results can appear immediately. Otherwise ChurnIQ will start background training.`;
}
function resetUpload(){
  uploadedFile=null;
  schemaAnalysis=null;
  document.getElementById('file-input').value='';
  document.getElementById('schema-page').style.display='none';
  document.getElementById('upload-section').style.display='block';
}
function renderSchemaAnalysis(result){
  const rows=(result.mappings||[]).map(item=>`
    <div class="schema-box"><strong>${escapeHtml(item.source)}</strong><span>→ ${escapeHtml(item.canonical)}</span><small>Confidence: ${(item.confidence*100).toFixed(0)}%</small></div>`).join('');
  const missing=result.missing_features||[];
  const warning = result.dataset_type === 'labeled'
    ? `${result.n_rows} customers · ${result.n_columns} columns · labeled dataset detected`
    : (missing.length
      ? `Missing required features: ${missing.join(', ')}`
      : `${result.n_rows} customers · ${result.n_columns} columns · prediction dataset`);
  document.getElementById('schema-warning').textContent=warning;
  document.getElementById('schema-grid').innerHTML=`
    <div class="schema-box" style="grid-column:1/-1"><h4>Detected features</h4>${rows||'<span class="muted">No compatible canonical features detected.</span>'}</div>
    <div class="schema-box"><strong>Derived features</strong><span>${(result.derived_features||[]).join(', ')||'None'}</span></div>
    <div class="schema-box"><strong>Compatibility</strong><span>${(result.compatibility_score*100).toFixed(0)}%</span></div>
    ${result.model ? `<div class="schema-box"><strong>Compatible model</strong><span>${escapeHtml(result.model.model_name)} · AUC ${escapeHtml(result.model.auc_cv ?? '—')}</span></div>` : ''}`;
  const button = document.getElementById('confirm-btn');
  button.classList.remove('is-loading');

  const isUnlabeled = result.dataset_type === 'unlabeled';
  const hasMissingRequired = missing.length > 0;
  const hasCompatibleModel = !!result.model;
  const compatibilityPct = Number(result.compatibility_score || 0) * 100;
  const compatibilityTier =
    compatibilityPct >= 95 ? 'Excellent' :
    compatibilityPct >= 85 ? 'Good' :
    compatibilityPct >= 70 ? 'Warning' : 'Low';

  // A dataset can also be compatible with the existing active (legacy)
  // model even if it does not use the canonical transaction feature names.
  // Do not block that safe backend compatibility check in the browser.
  const canPredictUnlabeled = isUnlabeled;

  const readinessText = isUnlabeled
    ? (hasCompatibleModel && !hasMissingRequired ? 'Ready to predict' : 'Ready to check model compatibility')
    : 'Ready for labeled analysis';
  const modelText = result.model
    ? `${escapeHtml(result.model.model_name)} (AUC ${escapeHtml(result.model.auc_cv ?? '—')})`
    : 'No compatible model selected yet';

  document.getElementById('schema-grid').innerHTML = `
    <div class="schema-box" style="grid-column:1/-1">
      <h4>Dataset Analysis</h4>
      <strong>${escapeHtml(readinessText)}</strong>
      <p class="muted" style="margin-top:8px">Type: ${escapeHtml(result.dataset_type || 'unknown')} · Compatibility: ${compatibilityPct.toFixed(0)}% (${compatibilityTier})</p>
      <p class="muted" style="margin-top:6px">Model: ${modelText}</p>
    </div>
    <div class="schema-box" style="grid-column:1/-1"><h4>Detected features</h4>${rows||'<span class="muted">No compatible canonical features detected.</span>'}</div>
    <div class="schema-box"><strong>Derived features</strong><span>${(result.derived_features||[]).join(', ')||'None'}</span></div>
    <div class="schema-box"><strong>Missing required features</strong><span>${missing.length ? escapeHtml(missing.join(', ')) : 'None'}</span></div>
  `;

  if (isUnlabeled) {
    button.textContent = hasCompatibleModel && !hasMissingRequired
      ? 'Predict Churn'
      : 'Check & Predict';
    button.disabled = false;
  } else {
    button.textContent = 'Confirm & Analyse';
    button.disabled = false;
  }
}
async function saveAndGo(payload, trainedOnUpload = false, trainingResult = null) {
  console.log('STEP 1: saveAndGo called');
  console.log('Payload received:', payload);
  console.log('Payload customers:', payload?.customers?.length);
  try {
    console.log('STEP 2: saving analytics...');
    await saveAnalytics(payload, trainedOnUpload, trainingResult);
    console.log('STEP 3: saveAnalytics completed');
    const saved = await loadAnalytics();
    console.log('STEP 4: loaded after save:', saved);
    console.log('Saved customers:', saved?.payload?.customers?.length);
    if (!saved || !saved.payload) {
      throw new Error('Analytics payload was not saved to browser storage.');
    }
    console.log('STEP 5: redirecting to dashboard');
    window.location.href = 'index.html';
  } catch (e) {
    console.error('saveAndGo ERROR:', e);
    alert(`Could not open the dashboard.\n\n${e.message}\n\nPlease try again.`);
  }
}
function showTraining(on){
  document.getElementById('training-overlay').classList.toggle('active',on);
  if(on){window.__trainStartedAt=Date.now();renderTrainSteps(0);updateTrainElapsed();}
  else{window.__trainStartedAt=null;}
}
const TRAIN_STAGES=['Dataset received','Feature engineering','Model training','Evaluation & scoring','Finalising'];
function stageIndexFromStep(step,pct){
  const s=String(step||'').toLowerCase();
  if(pct>=98||/finalis|finaliz|finish|complet|saving|persist/.test(s))return 4;
  if(/eval|score|metric|shap|valid|auc/.test(s))return 3;
  if(/train|fit|xgb|model|hyper|tuning/.test(s))return 2;
  if(/feature|engineer|rfm|prepar|clean|schema|column/.test(s))return 1;
  return 0;
}
function renderTrainSteps(activeIdx){
  const el=document.getElementById('train-steps');if(!el)return;
  el.innerHTML=TRAIN_STAGES.map((label,i)=>{
    const state=i<activeIdx?'done':(i===activeIdx?'active':'');
    const mark=i<activeIdx?'✓':String(i+1);
    return `<li class="${state}"><span class="dot" aria-hidden="true">${mark}</span><span>${label}</span></li>`;
  }).join('');
}
function updateTrainElapsed(){
  const el=document.getElementById('train-elapsed');if(!el)return;
  const started=window.__trainStartedAt;if(!started){el.textContent='';return;}
  const secs=Math.max(0,Math.round((Date.now()-started)/1000));
  el.textContent=(secs>=60?Math.floor(secs/60)+'m '+(secs%60)+'s':secs+'s')+' elapsed';
}
function resetProgress(){
  document.getElementById('train-progress-fill').style.width='0%';
  document.getElementById('train-progress-pct').textContent='0%';
  document.getElementById('train-step').textContent='Initialising...';
  const elapsed=document.getElementById('train-elapsed');
  if(elapsed)elapsed.textContent='0s elapsed';
  renderTrainSteps(0);
}
function setConfirmBusy(label){
  const button=document.getElementById('confirm-btn');
  if(!button)return;
  if(label){
    button.disabled=true;button.classList.add('is-loading');
    button.innerHTML=`<span class="btn-spinner" aria-hidden="true"></span>${escapeHtml(label)}`;
  }else{
    button.disabled=false;button.classList.remove('is-loading');
    button.textContent=schemaAnalysis?.dataset_type==='unlabeled'?'Predict Churn':'Confirm & Analyse';
  }
}
async function confirmAndAnalyze(){
  if(!uploadedFile)return;
  setConfirmBusy(schemaAnalysis?'Preparing analysis…':'Analyzing dataset…');
  try{
    if(!schemaAnalysis){
      document.getElementById('schema-grid').innerHTML =
        '<div class="schema-box skeleton" style="grid-column:1/-1;min-height:120px" aria-hidden="true"></div>' +
        '<div class="schema-box skeleton" style="min-height:92px" aria-hidden="true"></div>' +
        '<div class="schema-box skeleton" style="min-height:92px" aria-hidden="true"></div>';
      try {
        const response=await analyzeSchema(uploadedFile);
        schemaAnalysis=await response.json();
      } catch (error) {
        // If schema preview fails, fall back to legacy analyze for labeled flow.
        const fallback = await analyzeFile(uploadedFile);
        if (fallback.status===200){
          const payload=await fallback.json();
          await saveAndGo(payload, payload.trained_on_upload || false, payload.training_result || null);
          return;
        }
        if (fallback.status===202){
          const job=await fallback.json();
          showTraining(true);
          await pollTrainingJob(job.job_id,job.label);
          return;
        }
        throw error;
      }
      renderSchemaAnalysis(schemaAnalysis);
      return;
    }
    // Canonical models use the dedicated endpoint. Other unlabeled data is
    // checked by the established batch endpoint, which only scores an exact
    // legacy-model schema and never trains an unlabeled dataset.
    const useCanonicalPrediction = schemaAnalysis.dataset_type === 'unlabeled'
      && !!schemaAnalysis.model
      && !(schemaAnalysis.missing_features || []).length;
    const resp = useCanonicalPrediction
      ? await predictUnlabeled(uploadedFile)
      : await analyzeFile(uploadedFile);
    if(resp.status===200){
      const payload=await resp.json();
      await saveAndGo(payload, payload.trained_on_upload || false, payload.training_result || null);
      return;
    }
    if(resp.status===202){
      const job=await resp.json();
      showTraining(true);
      await pollTrainingJob(job.job_id,job.label);
      return;
    }
    let err={};try{err=await resp.json()}catch(_){}
    throw new Error(err.detail||'Backend unavailable.');
  }catch(e){
    showTraining(false);
    alert(`Could not analyse the dataset.\n\n${e.message}\n\nMake sure the backend is running on port 8000.`);
    setConfirmBusy(null);
  }
}
async function pollTrainingJob(jobId,label){
  document.getElementById('train-title').textContent=`Training model on "${label}"`;
  let connectionFailures=0;
  while(true){
    await new Promise(r=>setTimeout(r,1500));
    let status;
    try{
      status=await getTrainingStatus(jobId);
      connectionFailures=0;
    }catch(_){
      connectionFailures+=1;
      if(connectionFailures>=5) throw new Error('Backend unavailable.');
      continue;
    }
    document.getElementById('train-progress-fill').style.width=`${status.pct||0}%`;
    document.getElementById('train-progress-pct').textContent=`${status.pct||0}%`;
    document.getElementById('train-step').textContent=status.step||'Training...';
    renderTrainSteps(stageIndexFromStep(status.step,status.pct||0));
    updateTrainElapsed();
    if(status.status==='error'){throw new Error(status.error||'Training failed.')}
    if(status.status==='complete')break;
  }
  const payload=await getTrainingResult(jobId);
  showTraining(false);
  saveAndGo(payload, payload.trained_on_upload || false, payload.training_result || null);
}
function initUpload(){
  basePage('upload',`
    <div class="upload-layout">
      <section id="upload-section">
        <div class="upload-hero"><div class="icon">📊</div><h1>Analyse a new dataset</h1><p>Upload a CSV or Excel customer dataset. ChurnIQ detects the schema, reuses a compatible model when possible, or trains a new model automatically.</p></div>
        <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <div class="upload-icon">☁️</div><h2>Drop your dataset here</h2><p>or click to browse<br/>Supported formats: CSV, XLSX, XLS</p>
          <input type="file" id="file-input" accept=".csv,.xlsx,.xls"/>
        </div>
        <div class="hint-grid">
          <div class="hint"><strong>Automatic schema detection</strong><span>Numeric, categorical and ID columns are detected automatically.</span></div>
          <div class="hint"><strong>Churn target detection</strong><span>The backend identifies the churn target from the dataset.</span></div>
          <div class="hint"><strong>Model reuse</strong><span>Matching schemas can reuse an existing trained model.</span></div>
          <div class="hint"><strong>Explainability</strong><span>Churn predictions include SHAP-based driver information when available.</span></div>
        </div>
      </section>
      <section class="schema-page" id="schema-page" style="display:none">
        <div class="card schema-card">
          <div class="page-head" style="margin-bottom:0"><div><h1>Confirm Dataset</h1><p id="schema-subtitle">Review the detected file before analysis.</p></div></div>
          <div id="schema-warning" class="schema-warning"></div>
          <div class="schema-grid" id="schema-grid"></div>
          <div class="schema-actions"><button class="btn btn-secondary" onclick="resetUpload()">← Re-upload</button><button class="btn btn-primary" id="confirm-btn" onclick="confirmAndAnalyze()">✓ Confirm & Analyse</button></div>
        </div>
      </section>
    </div>
    <div class="training-overlay" id="training-overlay"><div class="train-card"><div class="brain">🧠</div><h2 id="train-title">Training model</h2><p>ChurnIQ is learning patterns from your dataset. Live progress is reported by the backend below.</p><ol class="train-steps" id="train-steps" aria-live="polite"></ol><div class="progress-track"><div class="progress-fill" id="train-progress-fill"></div></div><div class="train-meta"><span id="train-step">Initialising...</span><span id="train-elapsed">0s elapsed</span></div><div class="progress-pct" id="train-progress-pct">0%</div></div></div>
  `);
  setupDrop();
}
document.addEventListener('DOMContentLoaded',initUpload);
