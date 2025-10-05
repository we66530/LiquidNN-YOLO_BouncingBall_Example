import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import glob, os

# =========================
# 1) Define Models
# =========================
class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta  = nn.Parameter(torch.ones(hidden_size))
    def forward(self, x, h):
        dh = self.alpha * (torch.tanh(self.w_in(x) + self.w_h(h)) - h)
        return h + self.beta * dh

class LNNMotion(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super().__init__()
        self.cell = LiquidCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

class LSTMMotion(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # predict next (cx,cy)

# =========================
# 2) Dataset extraction (YOLO centers)
# =========================
yolo = YOLO("yolov8n.pt")

def extract_center_trajectories(video_path, seq_len=10):
    cap = cv2.VideoCapture(video_path)
    trajectories = []
    tracker = {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = yolo.track(frame, persist=True, verbose=False)
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for obj_id, box in zip(ids, boxes):
                x1,y1,x2,y2 = box
                cx = (x1+x2)/2.0/width
                cy = (y1+y2)/2.0/height
                if obj_id not in tracker:
                    tracker[obj_id] = []
                tracker[obj_id].append([cx, cy])
                if len(tracker[obj_id]) >= seq_len+1:
                    seq = tracker[obj_id][-seq_len-1:]
                    trajectories.append(seq)
    cap.release()
    return np.array(trajectories)

# =========================
# 3) Load training videos
# =========================
dataset_folder = r"C:\Users\User\Desktop\常用程式\ball_dataset"
seq_len = 10
all_data = []

for video_file in glob.glob(os.path.join(dataset_folder, "*.mp4")):
    print("Extracting:", video_file)
    data = extract_center_trajectories(video_file, seq_len)
    if len(data)>0:
        all_data.append(data)

if len(all_data) == 0:
    raise RuntimeError("❌ No training data found. Check YOLO detections.")

all_data = np.concatenate(all_data, axis=0)
X = all_data[:,:seq_len,:]   # [N, seq_len, 2]
Y = all_data[:,seq_len,:]    # [N, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

print("✅ Training dataset:", X_tensor.shape, Y_tensor.shape)

# =========================
# 4) Train function
# =========================
def train_model(model, X, Y, epochs=100, name="model"):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(epochs), desc=f"Training {name}"):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        opt.step()
        if (epoch+1)%10==0:
            print(f"[{name}] Epoch {epoch+1}, Loss={loss.item():.6f}")
    return model

# Train both models
lnn = LNNMotion().to(device)
lstm = LSTMMotion().to(device)

lnn = train_model(lnn, X_tensor, Y_tensor, epochs=100, name="LNN")
lstm = train_model(lstm, X_tensor, Y_tensor, epochs=100, name="LSTM")

torch.save(lnn.state_dict(), "lnn_motion_xy.pth")
torch.save(lstm.state_dict(), "lstm_motion_xy.pth")

# =========================
# 5) Metrics
# =========================
def mse_rmse(pred, true):
    mse = np.mean((pred-true)**2)
    rmse = np.sqrt(mse)
    return mse, rmse

def ade_fde(pred_seq, true_seq):
    diff = np.linalg.norm(pred_seq-true_seq, axis=1)
    return np.mean(diff), diff[-1]

def evaluate_on_video(video_path, model, name, seq_len=10):
    cap = cv2.VideoCapture(video_path)
    tracker = {}
    metrics = []
    width = int(cap.get(3))
    height = int(cap.get(4))
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = yolo.track(frame, persist=True, verbose=False)
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for obj_id, box in zip(ids, boxes):
                x1,y1,x2,y2 = box
                cx = (x1+x2)/2.0/width
                cy = (y1+y2)/2.0/height
                if obj_id not in tracker:
                    tracker[obj_id] = []
                tracker[obj_id].append([cx, cy])
                if len(tracker[obj_id]) >= seq_len+1:
                    seq = tracker[obj_id][-seq_len-1:]
                    X_seq = torch.tensor([seq[:seq_len]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pred = model(X_seq).cpu().numpy()[0]
                    true = seq[seq_len]
                    mse, rmse = mse_rmse(pred,true)
                    ade, fde = ade_fde(np.array(seq[:seq_len]), np.array(seq[1:seq_len+1]))
                    metrics.append([mse,rmse,ade,fde])
    cap.release()
    if len(metrics)>0:
        metrics = np.array(metrics)
        print(f"📊 {name} Evaluation on {video_path}")
        print("MSE:", metrics[:,0].mean())
        print("RMSE:", metrics[:,1].mean())
        print("ADE:", metrics[:,2].mean())
        print("FDE:", metrics[:,3].mean())
    else:
        print(f"⚠️ No trajectories found for {name}")

# =========================
# 6) Side-by-side evaluation
# =========================
test_video = r"C:\Users\User\Desktop\LNN\test.mp4"
evaluate_on_video(test_video, lnn,  "LNN",  seq_len=10)
evaluate_on_video(test_video, lstm, "LSTM", seq_len=10)
