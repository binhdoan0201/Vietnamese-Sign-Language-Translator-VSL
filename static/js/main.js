/**
 * static/js/main.js
 * =================
 * Client web nhận diện ngôn ngữ ký hiệu (VSL) thời gian thực.
 *
 * CẢI TIẾN HIỆU NĂNG & ĐỘ TRỄ (B2):
 * 1. Giảm kích thước ảnh gửi lên: 320x240, JPEG quality 0.6 (giảm ~70% dung lượng payload).
 * 2. Gửi kèm mốc thời gian thực performance.now() / 1000.0 (giây) theo từng frame.
 * 3. Cơ chế ACK (Flow Control): Client chỉ gửi frame tiếp theo khi Server đã xử lý xong
 *    frame trước, triệt tiêu hoàn toàn hiện tượng nghẽn hàng đợi (queue buildup).
 * 4. Tần suất gửi tối đa ~30 FPS (interval 33ms), tự động thích ứng với tốc độ xử lý của server.
 */

const socket = io();

const videoEl = document.getElementById('video');
const overlayEl = document.getElementById('overlay');
const overlayCtx = overlayEl.getContext('2d');

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const stageHint = document.getElementById('stageHint');
const bufferBar = document.getElementById('bufferBar');

const translationText = document.getElementById('translationText');
const translationConf = document.getElementById('translationConf');
const confidenceList = document.getElementById('confidenceList');
const soundToggle = document.getElementById('soundToggle');
const logEl = document.getElementById('log');

// Cấu hình khung hình gửi lên server (Task 1.2: hoàn nguyên về chuẩn 640x480 khớp lúc train)
const SEND_WIDTH = 640;
const SEND_HEIGHT = 480;
const JPEG_QUALITY = 0.7;
const TARGET_INTERVAL_MS = 33; // Tối đa ~30 FPS
const ACK_TIMEOUT_MS = 500;    // Timeout an toàn phòng trường hợp rớt gói

const MAX_LOG_LINES = 6;

let config = null;          // nạp từ /api/config
let captureCanvas = null;   // canvas ẩn 320x240 để chụp frame
let captureCtx = null;
let sendTimer = null;

// Biến điều khiển luồng ACK
let isWaitingAck = false;
let lastSendTime = 0;

// ============================================================
// LOG TIỆN ÍCH
// ============================================================
function log(message) {
  const p = document.createElement('p');
  p.textContent = `> ${message}`;
  logEl.appendChild(p);
  while (logEl.children.length > MAX_LOG_LINES) {
    logEl.removeChild(logEl.firstChild);
  }
  logEl.scrollTop = logEl.scrollHeight;
}

// ============================================================
// 1. TẢI CẤU HÌNH & DỰNG UI DANH SÁCH ĐỘ TIN CẬY
// ============================================================
async function loadConfig() {
  const res = await fetch('/api/config');
  config = await res.json();

  confidenceList.innerHTML = '';
  config.actions.forEach(action => {
    const li = document.createElement('li');
    li.className = 'confidence-row';
    li.dataset.action = action;
    li.innerHTML = `
      <span class="confidence-row__label">${config.dictionary[action] || action}</span>
      <span class="confidence-row__track"><span class="confidence-row__fill"></span></span>
      <span class="confidence-row__value">0%</span>
    `;
    confidenceList.appendChild(li);
  });

  log(`đã nạp ${config.actions.length} từ vựng (chuẩn hóa thời gian: ${config.target_duration || 3.2}s)`);
}

// ============================================================
// 2. KHỞI ĐỘNG WEBCAM
// ============================================================
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    videoEl.srcObject = stream;

    await new Promise(resolve => {
      videoEl.onloadedmetadata = resolve;
    });

    // Canvas ẩn kích thước 320x240 tối ưu cho xử lý
    captureCanvas = document.createElement('canvas');
    captureCanvas.width = SEND_WIDTH;
    captureCanvas.height = SEND_HEIGHT;
    captureCtx = captureCanvas.getContext('2d');

    // Canvas overlay khớp kích thước hiển thị thực tế
    resizeOverlay();
    window.addEventListener('resize', resizeOverlay);

    log('đã kết nối webcam (stream: 640x480 -> gửi: 320x240)');
    stageHint.textContent = 'Đưa tay vào khung hình để bắt đầu';

    startSendingFrames();
  } catch (err) {
    log(`lỗi webcam: ${err.message}`);
    stageHint.textContent = 'Không thể truy cập webcam - kiểm tra quyền trình duyệt';
  }
}

function resizeOverlay() {
  const rect = videoEl.getBoundingClientRect();
  overlayEl.width = rect.width;
  overlayEl.height = rect.height;
}

// ============================================================
// 3. GỬI FRAME KÈM CƠ CHẾ ACK (FLOW CONTROL)
// ============================================================
function startSendingFrames() {
  if (sendTimer) clearInterval(sendTimer);

  sendTimer = setInterval(() => {
    if (!socket.connected || videoEl.readyState < 2) return;

    const now = performance.now();

    // Nếu đang chờ ACK từ server nhưng quá timeout -> giải phóng để không bị treo
    if (isWaitingAck && (now - lastSendTime > ACK_TIMEOUT_MS)) {
      isWaitingAck = false;
    }

    // Nếu server vẫn đang bận xử lý frame trước -> bỏ qua frame này (drop frame tự nhiên)
    if (isWaitingAck) return;

    // Vẽ và nén ảnh về 320x240, chất lượng 0.6
    captureCtx.drawImage(videoEl, 0, 0, SEND_WIDTH, SEND_HEIGHT);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);

    isWaitingAck = true;
    lastSendTime = now;

    socket.emit('video_frame', {
      image: dataUrl,
      timestamp: now / 1000.0 // Mốc thời gian chính xác tính theo giây
    });
  }, TARGET_INTERVAL_MS);
}

// ============================================================
// 4. VẼ KHUNG XƯƠNG (SKELETON OVERLAY)
// ============================================================
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

function drawHand(points, color) {
  if (!points) return;
  const w = overlayEl.width, h = overlayEl.height;

  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 1.5;
  overlayCtx.beginPath();
  HAND_CONNECTIONS.forEach(([a, b]) => {
    const pa = points[a], pb = points[b];
    if (!pa || !pb) return;
    overlayCtx.moveTo(pa[0] * w, pa[1] * h);
    overlayCtx.lineTo(pb[0] * w, pb[1] * h);
  });
  overlayCtx.stroke();

  overlayCtx.fillStyle = color;
  points.forEach(p => {
    overlayCtx.beginPath();
    overlayCtx.arc(p[0] * w, p[1] * h, 2, 0, Math.PI * 2);
    overlayCtx.fill();
  });
}

function drawLandmarks(data) {
  overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
  drawHand(data.left_hand, '#5EEAD4');
  drawHand(data.right_hand, '#C9A227');
}

// ============================================================
// 5. SOCKET EVENTS
// ============================================================
socket.on('connect', () => {
  statusDot.classList.add('live');
  statusText.textContent = 'Đang hoạt động';
  isWaitingAck = false;
  log('đã kết nối tới server');
});

socket.on('disconnect', () => {
  statusDot.classList.remove('live');
  statusText.textContent = 'Mất kết nối';
  isWaitingAck = false;
  log('mất kết nối server');
});

// Nhận phản hồi ACK từ Server để cho phép gửi frame kế tiếp
socket.on('frame_ack', () => {
  isWaitingAck = false;
});

socket.on('landmarks', (data) => {
  drawLandmarks(data);
});

socket.on('buffer_progress', (data) => {
  bufferBar.style.width = `${Math.min(data.progress * 100, 100)}%`;

  if (data.cooldown) {
    stageHint.textContent = `Đang chờ... (${data.cooldown_left || ''}s)`;
  } else if (!data.has_hand) {
    stageHint.textContent = 'Đưa tay vào khung hình để bắt đầu';
  } else {
    stageHint.textContent = 'Đang thu thập chuyển động...';
  }
});

socket.on('probabilities', (data) => {
  const probs = data.probs;
  const topAction = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];

  Object.entries(probs).forEach(([action, value]) => {
    const row = confidenceList.querySelector(`[data-action="${action}"]`);
    if (!row) return;
    const pct = Math.round(value * 100);
    row.querySelector('.confidence-row__fill').style.width = `${pct}%`;
    row.querySelector('.confidence-row__value').textContent = `${pct}%`;
    row.classList.toggle('is-top', action === topAction);
  });
});

socket.on('prediction', (data) => {
  translationText.textContent = data.text;
  translationConf.textContent = `độ tin cậy ${Math.round(data.confidence * 100)}%`;
  log(`nhận diện: "${data.text}" (${Math.round(data.confidence * 100)}%)`);

  if (soundToggle.checked) {
    const audio = new Audio(data.audio_url);
    audio.play().catch(() => {});
  }
});

// ============================================================
// KHỞI ĐỘNG
// ============================================================
(async function init() {
  await loadConfig();
  await startCamera();
})();
