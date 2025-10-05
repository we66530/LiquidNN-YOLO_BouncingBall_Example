import cv2
import torch
import numpy as np
from ultralytics import YOLO

# =========================
# 1) Reload trained models
# =========================
class LiquidCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_in = torch.nn.Linear(input_size, hidden_size)
        self.w_h = torch.nn.Linear(hidden_size, hidden_size)
        self.alpha = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta  = torch.nn.Parameter(torch.ones(hidden_size))
    def forward(self, x, h):
        dh = self.alpha * (torch.tanh(self.w_in(x) + self.w_h(h)) - h)
        return h + self.beta * dh

class LNNMotion(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super().__init__()
        self.cell = LiquidCell(input_size, hidden_size)
        self.fc   = torch.nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

class LSTMMotion(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lnn = LNNMotion().to(device)
lstm = LSTMMotion().to(device)

lnn.load_state_dict(torch.load("lnn_motion_xy.pth", map_location=device))
lstm.load_state_dict(torch.load("lstm_motion_xy.pth", map_location=device))
lnn.eval()
lstm.eval()

# =========================
# 2) Rollout function (multi-step prediction)
# =========================
@torch.no_grad()
def rollout(model, history, future_steps, device, seq_len):
    seq = history[-seq_len:].copy()
    preds = []
    for _ in range(future_steps):
        X = torch.tensor([seq], dtype=torch.float32, device=device)
        pred = model(X).cpu().numpy()[0].tolist()
        preds.append(pred)
        seq.append(pred)
        seq = seq[-seq_len:]
    return preds

# =========================
# 3) Video generation function
# =========================
def make_video(video_path, model, out_name, seq_len=10, future_steps=5, fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    width, height = int(cap.get(3)), int(cap.get(4))
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

    yolo = YOLO("yolov8n.pt")
    centers_hist, size_hist = {}, {}

    while True:
        ret, frame = cap.read()
        if not ret: break
        results = yolo.track(frame, persist=True, verbose=False)

        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for obj_id, box in zip(ids, boxes):
                x1,y1,x2,y2 = map(int, box)
                w,h = x2-x1, y2-y1
                cx = (x1+x2)/2.0/width
                cy = (y1+y2)/2.0/height

                # store center + size
                if obj_id not in centers_hist:
                    centers_hist[obj_id] = []
                    size_hist[obj_id] = (w,h)
                centers_hist[obj_id].append([cx,cy])
                size_hist[obj_id] = (w,h)

                # draw YOLO (green)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # LNN/LSTM rollout predictions
                if len(centers_hist[obj_id]) >= seq_len:
                    preds = rollout(model, centers_hist[obj_id], future_steps, device, seq_len)
                    for k,(cxn,cyn) in enumerate(preds, start=1):
                        px,py = int(cxn*width), int(cyn*height)
                        w,h = size_hist[obj_id]
                        color = (0, min(255, 50+40*k), 255)  # red→orange→yellow
                        cv2.rectangle(frame, (px-w//2, py-h//2), (px+w//2, py+h//2), color, 2)
                        cv2.putText(frame, f"+{k}", (px-w//2, py-h//2-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved {out_name}")

# =========================
# 4) Run for both models
# =========================
test_video = r"C:\Users\User\Desktop\LNN\test.mp4"
make_video(test_video, lnn,  "YOLO_LNN.mp4",  seq_len=10, future_steps=5)
make_video(test_video, lstm, "YOLO_LSTM.mp4", seq_len=10, future_steps=5)
