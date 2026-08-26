const API = window.location.protocol === 'file:'
  ? 'http://127.0.0.1:8000'
  : `${window.location.protocol}//${window.location.hostname}:8000`;

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response;
}

async function getHealth() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3000);
  try {
    const response = await apiFetch('/health', { signal: controller.signal });
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

/* =========================================================
   IndexedDB analytics persistence
   ========================================================= */
const DB_NAME = 'churniq-db';
const DB_VERSION = 2; // v2 adds the applied-strategies store
const STORE_NAME = 'analytics';
const APPLIED_STORE = 'applied_strategies';

function openAnalyticsDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    const timeout = setTimeout(() => {
      reject(new Error('Browser storage did not respond. Please refresh the page.'));
    }, 5000);

    const finish = (callback, value) => {
      clearTimeout(timeout);
      callback(value);
    };

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
      if (!db.objectStoreNames.contains(APPLIED_STORE)) {
        db.createObjectStore(APPLIED_STORE);
      }
    };

    request.onsuccess = () => finish(resolve, request.result);
    request.onerror = () => finish(reject, request.error || new Error('Could not open analytics database.'));
    request.onblocked = () => finish(reject, new Error('Browser storage is locked by another ChurnIQ tab. Close other tabs and refresh.'));
  });
}

/* Applied retention-strategy tracking (persists across sessions) */
async function getAppliedStrategies() {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(APPLIED_STORE, 'readonly');
    const store = tx.objectStore(APPLIED_STORE);
    const valuesReq = store.getAll();
    const keysReq = store.getAllKeys();
    tx.oncomplete = () => {
      db.close();
      const out = {};
      (keysReq.result || []).forEach((key, i) => { out[key] = valuesReq.result[i]; });
      resolve(out);
    };
    tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not load applied strategies.')); };
  });
}
async function setStrategyState(key, applied) {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(APPLIED_STORE, 'readwrite');
    const store = tx.objectStore(APPLIED_STORE);
    if (applied) store.put({ appliedAt: Date.now() }, key);
    else store.delete(key);
    tx.oncomplete = () => { db.close(); resolve(true); };
    tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not update strategy state.')); };
  });
}

async function saveAnalytics(payload, trainedOnUpload = false, trainingResult = null) {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const value = {
      payload: payload,
      trainedOnUpload: !!trainedOnUpload,
      trainingResult: trainingResult || null,
      savedAt: Date.now()
    };
    const request = store.put(value, 'current');
    request.onerror = () => {
      console.error('IndexedDB PUT failed:', request.error);
      reject(request.error || new Error('Could not save analytics data.'));
    };
    transaction.oncomplete = () => {
      console.log('IndexedDB SAVE SUCCESS');
      db.close();
      resolve(true);
    };
    transaction.onerror = () => {
      console.error('IndexedDB transaction failed:', transaction.error);
      db.close();
      reject(transaction.error || new Error('IndexedDB transaction failed.'));
    };
    transaction.onabort = () => {
      console.error('IndexedDB transaction aborted:', transaction.error);
      db.close();
      reject(transaction.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

async function loadAnalytics() {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get('current');
    const timeout = setTimeout(() => {
      db.close();
      reject(new Error('Stored analytics took too long to load. Refresh the page and try again.'));
    }, 5000);
    const finish = (callback, value) => {
      clearTimeout(timeout);
      callback(value);
    };
    request.onsuccess = () => {
      const value = request.result;
      console.log(
        'IndexedDB LOAD:',
        value ? 'DATA FOUND' : 'NO DATA'
      );
      if (!value || !value.payload) {
        finish(resolve, null);
      } else {
        finish(resolve, {
          payload: value.payload,
          trainedOnUpload: !!value.trainedOnUpload,
          trainingResult: value.trainingResult || null
        });
      }
    };
    request.onerror = () => {
      console.error('IndexedDB GET failed:', request.error);
      finish(reject, request.error || new Error('Could not load analytics data.'));
    };
    transaction.oncomplete = () => {
      db.close();
    };
    transaction.onerror = () => {
      db.close();
    };
  });
}

async function clearAnalytics() {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    store.delete('current');
    transaction.oncomplete = () => {
      db.close();
      resolve(true);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error('Could not clear analytics data.'));
    };
  });
}

async function hasAnalytics() {
  const data = await loadAnalytics();
  return !!data;
}

async function saveCustomerFocus(customerId) {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put({ customerId, savedAt: Date.now() }, 'focus');

    request.onsuccess = () => resolve(true);
    request.onerror = () => reject(request.error || new Error('Could not save customer focus.'));
    transaction.oncomplete = () => db.close();
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error('Could not save customer focus.'));
    };
  });
}

async function loadCustomerFocus() {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const request = transaction.objectStore(STORE_NAME).get('focus');

    request.onsuccess = () => resolve(request.result?.customerId || null);
    request.onerror = () => reject(request.error || new Error('Could not load customer focus.'));
    transaction.oncomplete = () => db.close();
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error('Could not load customer focus.'));
    };
  });
}

async function clearCustomerFocus() {
  const db = await openAnalyticsDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).delete('focus');

    transaction.oncomplete = () => {
      db.close();
      resolve(true);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error('Could not clear customer focus.'));
    };
  });
}

/* =========================================================
   Backend analysis
   ========================================================= */
async function analyzeSchema(file) {
  const formData = new FormData();
  formData.append('file', file);
  return apiFetch('/schema/analyze', { method: 'POST', body: formData });
}

async function analyzeFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  return fetch(
    `${API}/predict/batch-analyze`,
    { method: 'POST', body: formData }
  );
}

async function predictUnlabeled(file) {
  const formData = new FormData();
  formData.append('file', file);
  return apiFetch('/predict/unlabeled', { method: 'POST', body: formData });
}

async function getTrainingStatus(jobId) {
  const response = await apiFetch(`/train/status/${jobId}`);
  return response.json();
}

async function getTrainingResult(jobId) {
  const response = await apiFetch(`/train/result/${jobId}`);
  return response.json();
}

async function analyzeEcommerceFile(file) { const formData = new FormData(); formData.append('file', file); const response = await apiFetch('/integrations/ecommerce/analyze', {method: 'POST', body: formData}); return response.json(); }
async function importEcommerceFile(file) { const formData = new FormData(); formData.append('file', file); const response = await apiFetch('/integrations/ecommerce/import', {method: 'POST', body: formData}); return response.json(); }
async function getEcommerceSummary() { const response = await apiFetch('/integrations/ecommerce/summary'); return response.json(); }
async function predictEcommerceCustomers() { const response = await apiFetch('/integrations/ecommerce/predict', {method: 'POST'}); return response.json(); }
async function getPlatformStatus() { const response = await apiFetch('/integrations/platform/status'); return response.json(); }
async function syncPlatform(platform) { const response = await apiFetch(`/integrations/platform/sync/${platform}`, {method: 'POST'}); return response.json(); }
async function predictPlatform(platform) { const response = await apiFetch(`/integrations/platform/predict/${platform}`, {method: 'POST'}); return response.json(); }
/* Applied retention strategies — server-side truth so state reflects on every device */
async function fetchAppliedStrategies() { const response = await apiFetch('/strategies/applied'); return response.json(); }
async function applyStrategyOnServer(strategyKey, customerIds, action, applied) {
  const response = await apiFetch('/strategies/apply', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({customer_ids: customerIds, strategy_key: strategyKey, action, applied})
  });
  return response.json();
}
/* Retention outreach — email / WhatsApp the applied strategy to a customer */
async function notifyStrategy(customerId, strategyKey, action, channel, phone) {
  const response = await apiFetch('/strategies/notify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({customer_id: customerId, strategy_key: strategyKey, action, channel, phone})
  });
  return response.json();
}
async function getNotificationOutbox() {
  const response = await apiFetch('/notifications/outbox');
  return response.json();
}
