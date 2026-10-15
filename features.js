// 얼굴 기준점(MediaPipe Face Mesh) -> 비율 특징값. 학습용 일괄처리와 웹페이지가 같은 코드를 씀.

// lm: [{x,y}] (0~1 정규화), w/h: 이미지 크기. 반환: {values:{...}, yaw}
function faceFeatures(lm, w, h) {
  const P = lm.map(p => [p.x * w, p.y * h]);
  // 두 눈 바깥 꼬리를 잇는 선이 수평이 되도록 회전
  const ang = Math.atan2(P[263][1] - P[33][1], P[263][0] - P[33][0]);
  const c = Math.cos(-ang), s = Math.sin(-ang);
  const R = P.map(([x, y]) => [x * c - y * s, x * s + y * c]);
  const d = (a, b) => Math.hypot(R[a][0] - R[b][0], R[a][1] - R[b][1]);
  const W = d(234, 454), H = d(10, 152);
  const eyeW = (d(33, 133) + d(263, 362)) / 2;
  const tilt = (o, i, sign) => Math.atan2(R[i][1] - R[o][1], sign * (R[o][0] - R[i][0])) * 180 / Math.PI;
  const values = {
    face_ratio: H / W,
    eye_w: eyeW / W,
    eye_open: (d(159, 145) + d(386, 374)) / 2 / eyeW,
    eye_gap: d(133, 362) / W,
    eye_y: ((R[159][1] + R[386][1]) / 2 - R[10][1]) / H,
    eye_tilt: (tilt(33, 133, -1) + tilt(263, 362, 1)) / 2,
    brow_gap: (d(105, 159) + d(334, 386)) / 2 / H,
    nose_len: d(168, 2) / H,
    nose_w: d(129, 358) / W,
    mouth_w: d(61, 291) / W,
    lip: d(0, 17) / H,
    lower: d(2, 152) / H,
    jaw_w: d(172, 397) / W,
  };
  // 고개 돌린 정도: 코끝에서 좌우 얼굴 끝까지 거리 차이 (0 = 정면)
  const yaw = Math.abs(d(1, 234) - d(1, 454)) / W;
  return { values, yaw };
}

// 얼굴이 작게 찍힌 사진도 잡도록 전체 -> 위쪽 일부를 잘라 확대하며 재시도
const CROPS = [[0,0,1,1],[0.1,0,0.9,0.6],[0.2,0,0.8,0.45],[0.25,0.05,0.75,0.4],[0,0,0.6,0.5],[0.4,0,1,0.5]];
async function detectFace(fm, img) {
  const W = img.naturalWidth, H = img.naturalHeight;
  for (const [x0, y0, x1, y1] of CROPS) {
    const cv = document.createElement('canvas');
    const sw = (x1 - x0) * W, sh = (y1 - y0) * H, k = 640 / Math.max(sw, sh);
    cv.width = Math.round(sw * k); cv.height = Math.round(sh * k);
    cv.getContext('2d').drawImage(img, x0 * W, y0 * H, sw, sh, 0, 0, cv.width, cv.height);
    const r = await new Promise(res => { fm.onResults(res); fm.send({ image: cv }); });
    const lm = r.multiFaceLandmarks && r.multiFaceLandmarks[0];
    if (lm) return faceFeatures(lm, cv.width, cv.height);
  }
  return null;
}
