import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Reproducibility
# -----------------------------
RND = 42
random.seed(RND)
np.random.seed(RND)

# -----------------------------
# 2. CPU Scheduling Algorithms
# -----------------------------

def fcfs(burst_times, arrival_times):
    n = len(burst_times)
    start = [0] * n
    finish = [0] * n
    waiting = [0] * n
    turnaround = [0] * n

    start[0] = arrival_times[0]
    finish[0] = start[0] + burst_times[0]
    turnaround[0] = finish[0] - arrival_times[0]
    waiting[0] = turnaround[0] - burst_times[0]

    for i in range(1, n):
        start[i] = max(finish[i-1], arrival_times[i])
        finish[i] = start[i] + burst_times[i]
        turnaround[i] = finish[i] - arrival_times[i]
        waiting[i] = turnaround[i] - burst_times[i]

    avg_wait = np.mean(waiting)
    avg_turnaround = np.mean(turnaround)
    return avg_wait, avg_turnaround


def sjf(burst_times, arrival_times):
    n = len(burst_times)
    completed = [False] * n
    current_time = 0
    completed_processes = 0
    waiting, turnaround = [0]*n, [0]*n

    while completed_processes < n:
        idx = -1
        min_bt = float('inf')
        for i in range(n):
            if arrival_times[i] <= current_time and not completed[i]:
                if burst_times[i] < min_bt:
                    min_bt = burst_times[i]
                    idx = i

        if idx == -1:
            current_time += 1
            continue

        current_time += burst_times[idx]
        turnaround[idx] = current_time - arrival_times[idx]
        waiting[idx] = turnaround[idx] - burst_times[idx]
        completed[idx] = True
        completed_processes += 1

    avg_wait = np.mean(waiting)
    avg_turnaround = np.mean(turnaround)
    return avg_wait, avg_turnaround


def round_robin(burst_times, arrival_times, quantum=4):
    n = len(burst_times)
    remaining = burst_times[:]
    current_time = 0
    waiting, turnaround = [0]*n, [0]*n
    ready_queue = []
    completed = 0
    arrival_idx = 0

    while completed < n:
        while arrival_idx < n and arrival_times[arrival_idx] <= current_time:
            ready_queue.append(arrival_idx)
            arrival_idx += 1

        if not ready_queue:
            current_time += 1
            continue

        idx = ready_queue.pop(0)
        if remaining[idx] > quantum:
            current_time += quantum
            remaining[idx] -= quantum
        else:
            current_time += remaining[idx]
            remaining[idx] = 0
            turnaround[idx] = current_time - arrival_times[idx]
            waiting[idx] = turnaround[idx] - burst_times[idx]
            completed += 1

        while arrival_idx < n and arrival_times[arrival_idx] <= current_time:
            ready_queue.append(arrival_idx)
            arrival_idx += 1

        if remaining[idx] > 0:
            ready_queue.append(idx)

    avg_wait = np.mean(waiting)
    avg_turnaround = np.mean(turnaround)
    return avg_wait, avg_turnaround


# -----------------------------
# 3. Dataset Generation
# -----------------------------

def create_cpu_dataset(target_each=300):
    data = []
    counts = {"FCFS":0, "SJF":0, "RR":0}
    attempts = 0

    while min(counts.values()) < target_each and attempts < 20000:
        attempts += 1
        n_processes = random.randint(3, 10)
        burst_times = [random.randint(1, 20) for _ in range(n_processes)]
        arrival_times = sorted([random.randint(0, 10) for _ in range(n_processes)])
        quantum = random.randint(2, 6)

        f_wait, f_tat = fcfs(burst_times, arrival_times)
        s_wait, s_tat = sjf(burst_times, arrival_times)
        r_wait, r_tat = round_robin(burst_times, arrival_times, quantum)

        best = min([("FCFS", f_wait), ("SJF", s_wait), ("RR", r_wait)], key=lambda x: x[1])[0]

        if counts[best] < target_each:
            avg_bt = np.mean(burst_times)
            std_bt = np.std(burst_times)
            cpu_load = sum(burst_times) / (max(arrival_times) + 1)
            data.append([n_processes, avg_bt, std_bt, cpu_load, quantum, f_wait, s_wait, r_wait, best])
            counts[best] += 1

    df = pd.DataFrame(data, columns=[
        "num_processes","avg_burst","std_burst","cpu_load","quantum",
        "fcfs_wait","sjf_wait","rr_wait","best_algo"
    ])
    return df, counts


df, counts = create_cpu_dataset(target_each=250)
print("Class counts:", counts)
df.to_csv("cpu_scheduling_dataset.csv", index=False)
print("Saved: cpu_scheduling_dataset.csv")

# -----------------------------
# 4. Model Training
# -----------------------------

features = ["num_processes","avg_burst","std_burst","cpu_load","quantum"]
X = df[features]
y = df["best_algo"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=RND, stratify=y_enc)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RND),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RND),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RND)
}

results = {}
for name, model in models.items():
    if name in ["SVM", "LogisticRegression"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, acc)
    print(f"{name} accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("-"*40)

best_name = max(results, key=lambda k: results[k][1])
best_model = results[best_name][0]
print("Best model:", best_name, "with accuracy", results[best_name][1])

joblib.dump(le, "cpu_label_encoder.pkl")
joblib.dump(scaler, "cpu_scaler.pkl")
joblib.dump(best_model, "best_cpu_model.pkl")
print("Saved models: cpu_label_encoder.pkl, cpu_scaler.pkl, best_cpu_model.pkl")

# -----------------------------
# 5. Confusion Matrix Visualization
# -----------------------------
if best_name in ["SVM", "LogisticRegression"]:
    y_pred = best_model.predict(X_test_scaled)
else:
    y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - CPU Scheduling")
plt.savefig("cpu_confusion_matrix.png")
plt.show()

# -----------------------------
# 6. Prediction Function
# -----------------------------
def predict_cpu_sample(sample, use_scaled=False):
    df_input = pd.DataFrame([sample], columns=features)
    if use_scaled:
        arr = scaler.transform(df_input)
        pred = best_model.predict(arr)
    else:
        pred = best_model.predict(df_input)
    return le.inverse_transform(pred)[0]


print("Example predictions:")
print([5,10,3,7,4], "->", predict_cpu_sample([5,10,3,7,4], use_scaled=(best_name!="RandomForest")))
print([8,5,2,4,3], "->", predict_cpu_sample([8,5,2,4,3], use_scaled=(best_name!="RandomForest")))
