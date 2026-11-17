const byId = id => document.getElementById(id);
const token = new URLSearchParams(location.hash.slice(1)).get('token');
const canvas = byId('frame');
const context = canvas.getContext('2d');
const fields = ['hover', 'depth', 'yaw'].map(byId);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const busyStates = new Set(['planning', 'moving', 'stopping']);
let status;
let frameKey = '';
let frozen = false;
let posting = false;
let selectedPlanInputs = '';
let revision = 0;
let cursor;
let bitmap;

async function api(path, data) {
  const headers = { 'X-Grasp-Token': token || '' };
  const options = { headers, cache: 'no-store', signal: AbortSignal.timeout(3500) };
  if (data) {
    headers['Content-Type'] = 'application/json';
    options.method = 'POST';
    options.body = JSON.stringify(data);
  }
  const response = await fetch(`api/${path}`, options);
  if (!response.ok) throw new Error((await response.json()).error);
  return response;
}

function showError(message = '') {
  byId('error').textContent = message;
  byId('error').hidden = !message;
}

function formatPoint(point) {
  return point.map((value, i) => `${'XYZ'[i]} ${value.toFixed(1)}`).join(' · ') + ' mm';
}

function inputFingerprint() {
  return fields.map(field => field.value).join(',');
}

function render() {
  const busy = busyStates.has(status?.state);
  const hasTarget = Boolean(status?.target_mm);
  const currentPlan = status?.plan_id && selectedPlanInputs === inputFingerprint();
  byId('mode').textContent = status ? (status.live ? 'Motion enabled' : 'Preview mode') : 'Disconnected';
  byId('status').textContent = status?.message || 'Waiting for the local console.';
  byId('target').textContent = hasTarget ? formatPoint(status.target_mm) : 'Select a point on the image';
  byId('stop').disabled = !status;
  byId('clear').disabled = posting || busy || !hasTarget;
  byId('plan-hover').disabled = posting || busy || !hasTarget;
  byId('plan-grasp').disabled = posting || busy || !hasTarget || !status?.can_grasp;
  byId('confirm').disabled = posting || busy || !status?.live || !currentPlan;
  const steps = byId('steps');
  const poses = byId('poses');
  steps.hidden = !currentPlan || !status.steps.length;
  poses.hidden = !currentPlan;
  steps.replaceChildren();
  poses.replaceChildren();
  if (currentPlan) {
    for (const step of status.steps) {
      const item = document.createElement('li');
      item.textContent = step;
      steps.append(item);
    }
    for (const [label, point] of [['Hover', status.hover_mm], ['Grasp', status.grasp_mm]]) {
      if (!point) continue;
      const term = document.createElement('dt');
      const value = document.createElement('dd');
      term.textContent = label;
      value.textContent = formatPoint(point);
      poses.append(term, value);
    }
  }
}

async function post(action, data = {}) {
  if (posting && action !== 'stop') return false;
  posting = true;
  const requestRevision = ++revision;
  showError();
  render();
  try {
    const next = await (await api(action, data)).json();
    if (requestRevision !== revision) return false;
    status = next;
    return true;
  } catch (error) {
    if (requestRevision === revision) showError(error.message);
    return false;
  } finally {
    if (requestRevision === revision) {
      posting = false;
      render();
    }
  }
}

function resumeCamera() {
  frozen = false;
  frameKey = '';
  cursor = undefined;
}

function draw() {
  if (!bitmap) return;
  context.drawImage(bitmap, 0, 0);
  if (!cursor) return;
  const { x, y } = cursor;
  context.beginPath();
  context.arc(x, y, 9, 0, 2 * Math.PI);
  context.moveTo(x - 15, y);
  context.lineTo(x + 15, y);
  context.moveTo(x, y - 15);
  context.lineTo(x, y + 15);
  context.strokeStyle = '#17201c';
  context.lineWidth = 4;
  context.stroke();
  context.strokeStyle = '#9ae5c1';
  context.lineWidth = 2;
  context.stroke();
}

async function select(point) {
  if (!frameKey || posting || busyStates.has(status?.state)) return;
  frozen = true;
  cursor = point;
  draw();
  if (await post('target', { frame: frameKey, ...point })) {
    frameKey = '';
    byId('frame-status').textContent = 'Marked frame';
  } else {
    resumeCamera();
  }
}

canvas.addEventListener('click', event => {
  const bounds = canvas.getBoundingClientRect();
  select({
    x: Math.floor((event.clientX - bounds.left) * canvas.width / bounds.width),
    y: Math.floor((event.clientY - bounds.top) * canvas.height / bounds.height),
  });
});

canvas.addEventListener('keydown', event => {
  if (!frameKey) return;
  const deltas = { ArrowLeft: [-5, 0], ArrowRight: [5, 0], ArrowUp: [0, -5], ArrowDown: [0, 5] };
  if (!deltas[event.key] && event.key !== 'Enter') return;
  event.preventDefault();
  cursor ||= { x: Math.floor(canvas.width / 2), y: Math.floor(canvas.height / 2) };
  if (event.key === 'Enter') {
    select(cursor);
    return;
  }
  const [dx, dy] = deltas[event.key];
  cursor = {
    x: Math.min(canvas.width - 1, Math.max(0, cursor.x + dx)),
    y: Math.min(canvas.height - 1, Math.max(0, cursor.y + dy)),
  };
  draw();
});

for (const kind of ['hover', 'grasp']) {
  byId(`plan-${kind}`).addEventListener('click', () => {
    if (!fields.every(field => field.reportValidity())) return;
    selectedPlanInputs = inputFingerprint();
    post('plan', { kind, hover_mm: fields[0].valueAsNumber, depth_mm: fields[1].valueAsNumber, yaw_deg: fields[2].valueAsNumber });
  });
}
fields.forEach(field => {
  field.required = true;
  field.addEventListener('input', render);
});
byId('clear').addEventListener('click', async () => {
  if (await post('clear')) resumeCamera();
});
byId('confirm').addEventListener('click', async () => {
  if (status?.plan_id && await post('confirm', { plan_id: status.plan_id })) resumeCamera();
});
byId('stop').addEventListener('click', () => {
  resumeCamera();
  post('stop');
});

async function pollStatus() {
  while (true) {
    const before = revision;
    try {
      const next = await (await api('status')).json();
      if (!posting && before === revision) {
        const lostTarget = status?.target_mm && !next.target_mm;
        status = next;
        if (lostTarget || status.state === 'moving' || status.state === 'stopping') resumeCamera();
        render();
      }
    } catch (error) {
      if (!posting && before === revision) {
        status = undefined;
        render();
        showError(error.message);
      }
    }
    await pause(250);
  }
}

async function pollCamera() {
  while (true) {
    if (!frozen) {
      try {
        const response = await api('frame');
        const image = await createImageBitmap(await response.blob());
        if (frozen) {
          image.close();
        } else {
          bitmap?.close();
          bitmap = image;
          if (canvas.width !== image.width) canvas.width = image.width;
          if (canvas.height !== image.height) canvas.height = image.height;
          frameKey = response.headers.get('X-Frame-Key');
          draw();
          byId('frame-status').textContent = frameKey ? 'Click the target surface' : 'Waiting for a settled arm pose';
        }
      } catch (error) {
        frameKey = '';
        byId('frame-status').textContent = error.message;
      }
    }
    await pause(125);
  }
}

if (token) {
  pollStatus();
  pollCamera();
} else {
  showError('Open the local URL printed by revobots-arm-grasp.');
}
