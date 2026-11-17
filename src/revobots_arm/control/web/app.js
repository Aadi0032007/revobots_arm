"use strict";

const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const tabs = [...document.querySelectorAll("[data-tab]")];
const liveActions = document.querySelectorAll("#connect, #disconnect, #preview-joints, #preview-pose, #copy-pose, #start-teleop, #apply-grasp, #sample, #reset-eye, #solve-eye, #preview-policy");
const serialDefault = navigator.userAgent.includes("Windows") ? ["COM4", "COM5"] : ["/dev/ttyACM0", "/dev/ttyACM1"];
let tab = "setup";
let setup;
let state;
let imageUrl;
let imageCamera = "";
let sending = false;
let notice = "";
let resultKey = "";
let badPreview = "";
let teleopOwned = false;

async function api(path, body, binary = false) {
  const headers = { "X-Grasp-Token": token };
  const options = { headers, cache: "no-store" };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.method = "POST";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `Request failed (${response.status})`);
  }
  return binary ? response.blob() : response.json();
}

function showMessage(message, error = false) {
  $("message").textContent = message;
  $("message").classList.toggle("error", error || message.startsWith("Error:"));
}

async function action(path, body = {}) {
  if (sending && path !== "/api/stop") return false;
  sending = true;
  notice = "";
  try {
    await api(path, body);
    state = await api("/api/status");
    render(state);
    return true;
  } catch (error) {
    notice = error.message;
    showMessage(notice, true);
    return false;
  } finally {
    sending = false;
  }
}

function option(label, value) {
  return new Option(label, value);
}

function options(select, values, preferred = select.value) {
  const identity = JSON.stringify(values);
  if (select.dataset.options !== identity) {
    select.replaceChildren(...values.map(([label, value]) => option(label, value)));
    select.dataset.options = identity;
  }
  if (values.some(([, value]) => value === preferred)) select.value = preferred;
}

function field(labelText, current, change, spec = {}) {
  const label = document.createElement("label");
  const input = document.createElement(spec.options ? "select" : "input");
  if (spec.options) {
    options(input, spec.options, current);
  } else {
    input.type = spec.type || "text";
    if (input.type === "checkbox") {
      input.checked = Boolean(current);
      label.classList.add("check");
    } else {
      input.value = current ?? "";
      if (input.type === "number") input.step = spec.step || "any";
      if (spec.min !== undefined) input.min = spec.min;
      if (spec.max !== undefined) input.max = spec.max;
    }
  }
  input.required = spec.required || false;
  input.addEventListener("input", () => {
    let value = input.value;
    if (input.type === "number") value = input.value === "" ? null : input.valueAsNumber;
    if (input.type === "checkbox") value = input.checked;
    change(value);
  });
  if (input.type === "checkbox") label.append(input, labelText);
  else label.append(labelText, input);
  return label;
}

function setting(container, object, key, label, spec = {}) {
  $(container).append(field(label, object[key], (value) => {
    object[key] = value;
    summarizeDriver();
  }, spec));
}

function summarizeDriver() {
  $("advanced").value = JSON.stringify({ robot: setup.robot, teleop: setup.teleop, ik: setup.ik }, null, 2);
}

function renderRobot() {
  $("robot-fields").replaceChildren();
  const robot = setup.robot;
  $("follower").value = robot.type;
  setting("robot-fields", robot, "id", "Robot ID", { required: true });
  if (robot.type.startsWith("agx_nero")) {
    setting("robot-fields", robot, "channel", "CAN channel", { required: true });
    setting("robot-fields", robot, "interface", "CAN interface", { required: true });
    setting("robot-fields", robot, "firmware_version", "Firmware profile", { required: true });
    if (robot.type === "agx_nero_native") {
      setting("robot-fields", robot, "with_gripper", "Gripper attached", { type: "checkbox" });
      setting("robot-fields", robot, "gripper_mode", "Gripper mode", {
        options: [["Width (m)", "width"], ["Automatic (%)", "auto"], ["Angle (°)", "angle"]],
      });
    }
  } else {
    setting("robot-fields", robot, "port", "Serial port", { required: true });
    setting("robot-fields", robot, "use_degrees", "Joint positions in degrees", { type: "checkbox" });
  }
}

function renderInput() {
  const input = setup.teleop;
  const hint = $("input-hint");
  $("input").value = input.type;
  $("input-fields").replaceChildren();
  hint.textContent = "";
  if (input.type === "manual") return;
  if (input.type === "gamepad_ee") {
    $("input-fields").append(field("Gamepad connection", input.udp_port ? "udp" : "usb", (value) => {
      input.udp_port = value === "udp" ? 5003 : null;
      renderInput();
      summarizeDriver();
    }, { options: [["USB on controller machine", "usb"], ["Remote UDP", "udp"]] }));
    if (input.udp_port !== null && input.udp_port !== undefined) {
      setting("input-fields", input, "udp_port", "UDP listen port", { type: "number", min: 1, max: 65535, step: 1, required: true });
      hint.textContent = "Run revobots-arm-gamepad-client on the gamepad machine and point it to this machine’s IP and UDP port.";
    } else {
      hint.textContent = "Plug the gamepad into the machine running the console. Hold LT to move.";
    }
    return;
  }
  if (input.type.endsWith("_remote")) {
    setting("input-fields", input, "host", "Bind IP", { required: true });
    setting("input-fields", input, "port", "UDP listen port", { type: "number", min: 1, max: 65535, step: 1, required: true });
    hint.textContent = "Run revobots-arm-leader-client on the leader machine. Use this machine’s IP as its destination.";
  } else {
    setting("input-fields", input, "id", "Leader ID", { required: true });
    setting("input-fields", input, "port", "Leader serial port", { required: true });
    if (input.type.startsWith("so")) {
      setting("input-fields", input, "use_degrees", "Leader positions in degrees", { type: "checkbox" });
    }
  }
  $("input-fields").append(field("Leader mapping", setup.ik.enabled ? "ik" : "joints", (value) => {
    setup.ik.enabled = value === "ik";
    summarizeDriver();
  }, { options: [["Joint mapping", "joints"], ["Cartesian IK", "ik"]] }));
}

function updateWrist() {
  const choices = setup.cameras.filter((c) => c.name && c.enabled && c.use_depth && c.type === "intelrealsense");
  options($("wrist"), [["None", ""], ...choices.map((c) => [c.name, c.name])], setup.wrist);
  if (!choices.some((c) => c.name === setup.wrist)) setup.wrist = $("wrist").value = "";
}

function renderCameras() {
  $("camera-fields").replaceChildren(...setup.cameras.map((camera, index) => {
    const card = document.createElement("div");
    card.className = "camera-card";
    const header = document.createElement("div");
    header.className = "actions";
    const title = document.createElement("strong");
    title.textContent = camera.name || "Camera";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${camera.name || "camera"}`);
    remove.onclick = () => {
      setup.cameras.splice(index, 1);
      renderCameras();
    };
    header.append(title, remove);
    const grid = document.createElement("div");
    grid.className = "grid";
    function add(key, label, spec = {}) {
      grid.append(field(label, camera[key], (value) => {
        if (key === "type") {
          camera[key] = value;
          if (value === "opencv") camera.use_depth = false;
          renderCameras();
          return;
        }
        if (key === "name") {
          if (setup.wrist === camera.name) setup.wrist = value;
          title.textContent = value || "Camera";
        }
        camera[key] = key === "rotation" ? Number(value) : value;
        updateWrist();
      }, spec));
    }
    add("name", "Name", { required: true });
    add("type", "Driver", { options: [["USB / OpenCV", "opencv"], ["Intel RealSense", "intelrealsense"]] });
    add("source", camera.type === "opencv" ? "Device index or path" : "Device serial or name", { required: true });
    add("fps", "FPS", { type: "number", min: 1, max: 120, step: 1, required: true });
    add("width", "Width", { type: "number", min: 1, max: 4096, step: 1, required: true });
    add("height", "Height", { type: "number", min: 1, max: 4096, step: 1, required: true });
    add("rotation", "Rotation", { options: [0, 90, 180, 270].map((n) => [`${n}°`, n]) });
    add("enabled", "Enabled", { type: "checkbox" });
    if (camera.type === "intelrealsense") add("use_depth", "Capture depth", { type: "checkbox" });
    card.append(header, grid);
    return card;
  }));
  updateWrist();
}

function renderSetup() {
  $("preset").value = setup.preset;
  $("fps").value = setup.fps;
  renderRobot();
  renderInput();
  renderCameras();
  summarizeDriver();
}

async function loadPreset(name) {
  const data = await api(`/api/config${name ? `?preset=${encodeURIComponent(name)}` : ""}`);
  setup = data.setup;
  options($("preset"), data.presets.map((preset) => [preset, preset]));
  for (const [key, value] of Object.entries(data.policy)) $("policy-form").elements[key].value = value ?? "";
  renderSetup();
}

function number(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function positions(values) {
  const list = document.createElement("dl");
  list.className = "positions";
  for (const [label, value] of values) {
    const row = document.createElement("div");
    const title = document.createElement("dt");
    const data = document.createElement("dd");
    title.textContent = label;
    data.textContent = value;
    row.append(title, data);
    list.append(row);
  }
  return list;
}

function renderJoints(info) {
  const container = $("joint-fields");
  const names = info.joints.map(({ key }) => key).join(" ");
  if (container.dataset.keys !== names) {
    container.dataset.keys = names;
    container.replaceChildren(...info.joints.map((joint) => {
      const row = document.createElement("div");
      row.className = "joint-row";
      const measured = document.createElement("output");
      const target = field(`${joint.key.replace(/\.pos$/, "")} target (${joint.unit})`, "", () => {}, {
        type: "number", min: joint.min, max: joint.max,
      });
      target.querySelector("input").name = joint.key;
      row.append(measured, target);
      return row;
    }));
  }
  info.joints.forEach((joint, index) => {
    container.children[index].querySelector("output").textContent = `${number(joint.value)} ${joint.unit}`;
  });
  const unitValues = info.joints.map(({ key, value, unit }) => [key.replace(/\.pos$/, ""), `${number(value)} ${unit}`]);
  if (info.pose) unitValues.push(["XYZ (mm)", info.pose.xyz.map(number).join(", ")],
    ["RPY (°)", info.pose.rpy.map(number).join(", ")]);
  if (info.feedback_error || !unitValues.length) {
    $("measured").textContent = info.feedback_error || "Connect the follower to read position.";
  } else {
    $("measured").replaceChildren(positions(unitValues));
  }
}

function renderResult(info) {
  const preview = info.preview;
  const key = preview ? `${preview.id}.${info.live}.${info.job}.${badPreview}` : "";
  if (key === resultKey) return;
  resultKey = key;
  for (const kind of ["joints", "pose", "policy"]) $(`${kind}-preview`).hidden = true;
  if (!preview || badPreview === preview.id) return;
  const target = $(`${preview.kind}-preview`);
  const caption = document.createElement("p");
  caption.textContent = `${preview.kind === "policy" ? "First action" : "Final targets"} · ${number(preview.seconds)} s${info.live ? " · confirm within 20 s" : ""}`;
  const values = positions(Object.entries(preview.targets).map(([name, value]) => [name, number(value)]));
  target.replaceChildren(caption, values);
  if (info.live) {
    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "primary";
    confirm.textContent = preview.kind === "policy" ? "Run policy" : "Execute move";
    confirm.disabled = Boolean(info.job);
    confirm.onclick = () => action("/api/execute", { id: preview.id });
    target.append(confirm);
  }
  target.hidden = false;
}

function renderPick() {
  const frame = $("pick-frame");
  const visible = tab === "pick" && state?.grasp_ready;
  frame.hidden = !visible;
  if (visible && !frame.hasAttribute("src")) frame.src = `/grasp/#token=${encodeURIComponent(token)}`;
  if (!visible && frame.hasAttribute("src")) frame.removeAttribute("src");
}

function render(info) {
  if (!notice) showMessage(info.message);
  const idle = !info.job;
  const connectedIdle = idle && info.connected;
  $("mode").textContent = info.live ? "Live motion" : "Preview mode";
  $("mode").classList.toggle("live", info.live);
  $("connection").textContent = info.connected ? "Follower connected" : "Disconnected";
  $("connection").classList.toggle("online", info.connected);
  $("stop").disabled = !info.connected && idle;
  $("connect").disabled = !idle || info.connected;
  $("disconnect").disabled = !connectedIdle;
  $("preview-joints").disabled = !connectedIdle || !info.joint_control || !info.joints.length;
  $("preview-pose").disabled = !connectedIdle || !info.pose;
  $("preview-policy").disabled = !connectedIdle;
  $("copy-pose").disabled = !info.pose;
  $("rpy").hidden = !info.pose?.orientation;
  for (const input of $("rpy").querySelectorAll("input")) input.disabled = $("rpy").hidden;
  $("pose-base").textContent = info.pose
    ? `From ${info.pose.base} to ${info.pose.tip} · ${info.pose.orientation ? "Position and orientation" : "Position only"}`
    : "Tool pose is available for native Nero and SO-101 configured in degrees.";
  $("start-teleop").disabled = !connectedIdle || !info.live || info.input === "manual";
  $("teleop-status").textContent = info.input && info.input !== "manual"
    ? `Connected input: ${$("input").querySelector(`option[value="${info.input}"]`)?.textContent || info.input}.${info.live ? "" : " Launch with --execute to start teleop."}`
    : "Connect a follower and select a leader or gamepad in Setup.";
  if (info.job !== "teleop") teleopOwned = false;
  $("apply-grasp").disabled = !connectedIdle || info.robot !== "agx_nero_native" || !info.wrist;
  const cameras = info.cameras.map((name) => [name, name]);
  options($("preview-camera"), cameras);
  const eyeCamera = $("eye-form").elements.camera;
  options(eyeCamera, cameras, eyeCamera.value || info.wrist);
  $("sample").disabled = !connectedIdle || !info.cameras.length || info.robot !== "agx_nero_native";
  $("reset-eye").disabled = !connectedIdle || !info.eye_count;
  $("solve-eye").disabled = !connectedIdle || info.eye_count < 6;
  $("eye-count").textContent = `${info.eye_count} view${info.eye_count === 1 ? "" : "s"} captured`;
  $("use-eye").hidden = !info.eye_file;
  $("failed-cameras").hidden = !info.failed_cameras.length;
  $("failed-cameras").textContent = `Unavailable cameras: ${info.failed_cameras.join(", ")}`;
  renderJoints(info);
  renderResult(info);
  renderPick();
}

function selectTab(name) {
  tab = name;
  notice = "";
  if (state) showMessage(state.message);
  for (const item of tabs) {
    const active = item.dataset.tab === name;
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
    $(item.dataset.tab).hidden = !active;
  }
  $("monitor").hidden = name === "pick";
  document.querySelector(".layout").classList.toggle("in-grasp", name === "pick");
  renderPick();
}

function clearImage(message) {
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = "";
  $("camera-image").hidden = true;
  $("camera-image").removeAttribute("src");
  $("camera-state").hidden = false;
  $("camera-state").textContent = message;
}

async function updateImage() {
  const camera = $("preview-camera").value;
  if (document.hidden || tab === "pick") return;
  if (!camera) return clearImage("Connect a camera in Setup");
  if (imageCamera !== camera) clearImage("Waiting for camera…");
  imageCamera = camera;
  try {
    const frame = await api(`/api/frame?camera=${encodeURIComponent(camera)}`, undefined, true);
    if ($("preview-camera").value !== camera) return;
    const previous = imageUrl;
    imageUrl = URL.createObjectURL(frame);
    $("camera-image").src = imageUrl;
    $("camera-image").hidden = false;
    $("camera-state").hidden = true;
    if (previous) URL.revokeObjectURL(previous);
  } catch {
    clearImage("Waiting for a live camera frame");
  }
}

async function poll() {
  try {
    if (!sending) {
      state = await api("/api/status");
      render(state);
    }
  } catch (error) {
    showMessage(error.message, true);
    liveActions.forEach((button) => { button.disabled = true; });
  }
  setTimeout(poll, tab === "pick" ? 1000 : 400);
}

async function frames() {
  await updateImage();
  setTimeout(frames, 165);
}

function form(id, callback) {
  $(id).onsubmit = (event) => {
    event.preventDefault();
    callback(Object.fromEntries(new FormData(event.currentTarget)));
  };
}

tabs.forEach((item) => { item.onclick = () => selectTab(item.dataset.tab); });
$("preset").onchange = async () => {
  try {
    await loadPreset($("preset").value);
    notice = "";
  } catch (error) {
    showMessage(error.message, true);
  }
};
$("follower").onchange = () => {
  const type = $("follower").value;
  if (type.startsWith("agx_nero")) {
    const sameFamily = setup.robot.type.startsWith("agx_nero");
    const defaults = { id: "revobots_nero", interface: "socketcan", channel: "can-nero", firmware_version: "v121" };
    for (const key of Object.keys(defaults)) {
      if (sameFamily && setup.robot[key] !== undefined) defaults[key] = setup.robot[key];
    }
    setup.robot = type === "agx_nero_native"
      ? { type, ...defaults, with_gripper: false, gripper_mode: "width" }
      : { type, ...defaults, teleop_action_joints: ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex", "gripper"] };
  } else {
    setup.robot = { type, id: type, port: serialDefault[0], use_degrees: type.startsWith("so") };
  }
  if (setup.teleop.type === "gamepad_ee") {
    setup.teleop = { type: "gamepad_ee", udp_port: setup.teleop.udp_port ?? null };
    renderInput();
  }
  renderRobot();
  summarizeDriver();
};
$("input").onchange = () => {
  const type = $("input").value;
  if (type.endsWith("_remote")) {
    setup.teleop = { type, host: "0.0.0.0", port: type.startsWith("koch") ? 5001 : 5002 };
  } else if (type.endsWith("_leader")) {
    setup.teleop = { type, id: type, port: serialDefault[1], ...(type.startsWith("so") ? { use_degrees: true } : {}) };
  } else {
    setup.teleop = { type };
    setup.ik.enabled = false;
  }
  renderInput();
  summarizeDriver();
};
$("add-camera").onclick = () => {
  let index = setup.cameras.length + 1;
  while (setup.cameras.some((camera) => camera.name === `camera_${index}`)) index += 1;
  setup.cameras.push({ name: `camera_${index}`, source: 0, type: "opencv",
    width: 640, height: 480, fps: 30, rotation: 0, enabled: true, use_depth: false });
  renderCameras();
};
$("wrist").onchange = () => { setup.wrist = $("wrist").value; };
$("apply-advanced").onclick = () => {
  try {
    const advanced = JSON.parse($("advanced").value);
    if (advanced.robot?.type !== setup.robot.type || advanced.teleop?.type !== setup.teleop.type
        || typeof advanced.ik?.enabled !== "boolean") throw new Error("Use the Setup selectors to change device types.");
    Object.assign(setup, { robot: advanced.robot, teleop: advanced.teleop, ik: advanced.ik });
    renderSetup();
    notice = "";
  } catch (error) {
    notice = error.message;
    showMessage(notice, true);
  }
};
form("setup-form", () => {
  setup.fps = Number($("fps").value);
  const request = structuredClone(setup);
  for (const camera of request.cameras) {
    if (camera.type === "opencv" && /^\d+$/.test(String(camera.source))) camera.source = Number(camera.source);
  }
  action("/api/connect", request);
});
$("disconnect").onclick = () => action("/api/disconnect");
$("stop").onclick = () => action("/api/stop");
form("joints-form", (values) => action("/api/joints", {
  targets: Object.fromEntries(Object.entries(values).filter(([, value]) => value !== "").map(([key, value]) => [key, Number(value)])),
}));
const poseInputs = (id) => [...$(id).querySelectorAll("input")];
form("pose-form", () => action("/api/pose", {
  xyz: poseInputs("xyz").map((input) => input.valueAsNumber),
  rpy: $("rpy").hidden ? [0, 0, 0] : poseInputs("rpy").map((input) => input.valueAsNumber),
}));
$("copy-pose").onclick = () => {
  for (const id of ["xyz", "rpy"]) poseInputs(id).forEach((input, index) => {
    input.value = state.pose[id][index].toFixed(3);
  });
  $("pose-form").dispatchEvent(new Event("input"));
};
for (const kind of ["joints", "pose", "policy"]) {
  $(`${kind}-form`).oninput = () => {
    if (state?.preview?.kind === kind) {
      badPreview = state.preview.id;
      renderResult(state);
    }
  };
}
$("start-teleop").onclick = async () => {
  teleopOwned = await action("/api/teleop") && state?.job === "teleop";
};
window.addEventListener("pagehide", () => {
  if (!teleopOwned) return;
  fetch("/api/stop", { method: "POST", keepalive: true,
    headers: { "X-Grasp-Token": token, "Content-Type": "application/json" }, body: "{}" }).catch(() => {});
});
form("grasp-form", async (values) => {
  for (const key of ["table_z_m", "clearance_m", "gripper_open", "gripper_close"]) {
    values[key] = values[key] === "" ? null : Number(values[key]);
  }
  if (await action("/api/grasp-setup", values)) $("grasp-settings").open = false;
});
form("eye-form", (values) => action("/api/eye-sample", values));
form("solve-form", (values) => action("/api/eye-solve", values));
form("policy-form", (values) => action("/api/policy", values));
$("reset-eye").onclick = () => action("/api/eye-reset");
$("use-eye").onclick = () => {
  $("grasp-form").elements.calibration.value = state.eye_file;
  $("grasp-settings").open = true;
  selectTab("pick");
};
$("board").onclick = async () => {
  const { square, marker } = $("eye-form").elements;
  try {
    const png = await api(`/api/board?square=${encodeURIComponent(square.value)}&marker=${encodeURIComponent(marker.value)}`, undefined, true);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(png);
    link.download = "charuco-7x5.png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) {
    notice = error.message;
    showMessage(notice, true);
  }
};

async function initialize() {
  try {
    if (!token) throw new Error("Open the full console URL printed by revobots-arm-control.");
    await loadPreset();
    poll();
    frames();
  } catch (error) {
    showMessage(error.message, true);
    liveActions.forEach((button) => { button.disabled = true; });
  }
}

initialize();
