import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="Perimeter AI Prototype", layout="wide")
st.title("🛡️ AI Perimeter Safety – Intrusion & Loitering (Tracked + Zone-based)")

@st.cache_resource
def load_model():
    # yolov8n is fast; swap to yolov8s for better detection if needed
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_video = st.file_uploader("Upload warehouse CCTV video", type=["mp4", "avi", "mov"])

# Sidebar controls
st.sidebar.header("⚙️ Zone + Rules")

loiter_seconds = st.sidebar.slider("Loitering threshold (seconds)", 5, 120, 20)
miss_tolerance = st.sidebar.slider("Miss tolerance (seconds)", 0, 10, 2)
min_confidence = st.sidebar.slider("Detection confidence", 0.1, 1.0, 0.5, 0.05)

st.sidebar.subheader("🚫 Restricted Zone (Rectangle)")
zx1 = st.sidebar.slider("Zone x1", 0, 1920, 200)
zy1 = st.sidebar.slider("Zone y1", 0, 1080, 200)
zx2 = st.sidebar.slider("Zone x2", 0, 1920, 900)
zy2 = st.sidebar.slider("Zone y2", 0, 1080, 800)

show_trails = st.sidebar.checkbox("Show trails", value=True)

def inside_zone(cx, cy, x1, y1, x2, y2):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    return x_min <= cx <= x_max and y_min <= cy <= y_max

def draw_zone(frame, x1, y1, x2, y2):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (x_min + 10, max(30, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # UI layout
    colA, colB = st.columns([2, 1])
    with colA:
        stframe = st.empty()
        progress = st.progress(0)
    with colB:
        status_box = st.empty()
        metrics_box = st.empty()
        log_box = st.empty()
        debug_box = st.empty()

    # Tracking state per person_id
    # store: entered_once, entry_time, last_seen_in_zone, loiter_alerted, trail
    state = {}
    event_log = []

    frame_index = 0
    total_intrusions = 0
    total_loitering = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        t_sec = frame_index / fps

        if total_frames > 0:
            progress.progress(min(1.0, frame_index / total_frames))

        # Draw restricted zone
        draw_zone(frame, zx1, zy1, zx2, zy2)

        # --- IMPORTANT: built-in tracking for stable IDs ---
        results = model.track(frame, persist=True, conf=min_confidence, verbose=False)

        active_ids = set()
        persons_in_zone = 0

        # Extract tracked persons
        if results and len(results) > 0:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            if boxes is not None and boxes.id is not None:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    if cls != 0:  # only person
                        continue

                    person_id = int(boxes.id[i].item())
                    conf = float(boxes.conf[i].item())

                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    in_zone = inside_zone(cx, cy, zx1, zy1, zx2, zy2)
                    active_ids.add(person_id)

                    # init state
                    if person_id not in state:
                        state[person_id] = {
                            "entered_once": False,
                            "entry_time": None,
                            "last_seen_in_zone": None,
                            "loiter_alerted": False,
                            "trail": []
                        }

                    # update trail
                    state[person_id]["trail"].append((cx, cy))
                    if len(state[person_id]["trail"]) > 60:
                        state[person_id]["trail"].pop(0)

                    # update zone timestamps
                    if in_zone:
                        persons_in_zone += 1
                        state[person_id]["last_seen_in_zone"] = t_sec

                        # intrusion = entry event (first time inside after being outside)
                        if not state[person_id]["entered_once"]:
                            state[person_id]["entered_once"] = True
                            state[person_id]["entry_time"] = t_sec
                            state[person_id]["loiter_alerted"] = False

                            total_intrusions += 1
                            status_box.error(f"🚨 INTRUSION: Person {person_id} @ {t_sec:.1f}s")
                            event_log.append((t_sec, f"Person-{person_id}", "🚨 Intrusion Detected"))

                        # loitering check based on dwell from entry_time
                        if state[person_id]["entry_time"] is not None:
                            dwell = t_sec - state[person_id]["entry_time"]
                            if dwell >= loiter_seconds and not state[person_id]["loiter_alerted"]:
                                state[person_id]["loiter_alerted"] = True
                                total_loitering += 1
                                status_box.warning(f"⚠️ LOITERING: Person {person_id} ({int(dwell)}s) @ {t_sec:.1f}s")
                                event_log.append((t_sec, f"Person-{person_id}", f"⚠️ Loitering Detected ({int(dwell)}s)"))

                    else:
                        # If outside zone, we "arm" intrusion again only after a clean exit
                        # But don't instantly reset if we just missed detection briefly.
                        # Reset happens below using miss_tolerance logic.
                        pass

                    # choose color
                    if state[person_id]["loiter_alerted"]:
                        color = (0, 0, 255)
                        label = f"ID{person_id} LOITER"
                    elif state[person_id]["entered_once"]:
                        color = (0, 165, 255)
                        label = f"ID{person_id} INTR"
                    else:
                        color = (0, 255, 0)
                        label = f"ID{person_id}"

                    # draw person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # draw dwell if inside zone
                    if state[person_id]["entry_time"] is not None:
                        dwell = t_sec - state[person_id]["entry_time"]
                        cv2.putText(frame, f"Dwell: {int(dwell)}s", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # draw trail
                    if show_trails and len(state[person_id]["trail"]) > 1:
                        tr = state[person_id]["trail"]
                        for k in range(len(tr) - 1):
                            cv2.line(frame, tr[k], tr[k + 1], color, 2)

        # --- Miss tolerance reset logic ---
        # If a person hasn't been seen in-zone for more than miss_tolerance, treat as exited
        # and re-arm intrusion for next entry.
        to_reset = []
        for pid, s in state.items():
            if s["entered_once"]:
                last_in = s["last_seen_in_zone"]
                if last_in is None:
                    continue
                if (t_sec - last_in) > miss_tolerance:
                    to_reset.append(pid)

        for pid in to_reset:
            state[pid]["entered_once"] = False
            state[pid]["entry_time"] = None
            state[pid]["last_seen_in_zone"] = None
            state[pid]["loiter_alerted"] = False
            # keep trail

        # stats overlay
        cv2.putText(frame, f"Time: {int(t_sec)}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked IDs: {len(active_ids)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"In Zone: {persons_in_zone}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR", use_container_width=True)

        metrics_box.metric("Total Intrusions", total_intrusions)
        metrics_box.metric("Total Loitering", total_loitering)

        if event_log:
            txt = "📋 **Event Log (latest 10)**\n\n"
            for ts, who, evt in event_log[-10:]:
                txt += f"`{ts:>6.1f}s` | {who} | {evt}\n\n"
            log_box.markdown(txt)

        debug_box.info(
            f"Debug: fps={fps:.2f} | loiter={loiter_seconds}s | miss_tol={miss_tolerance}s"
        )

        time.sleep(1 / fps)

    cap.release()
    st.success("✅ Video processing complete!")

    st.subheader("📊 Summary")
    c1, c2 = st.columns(2)
    c1.metric("Total Intrusions Detected", total_intrusions)
    c2.metric("Total Loitering Events", total_loitering)

    if event_log:
        st.subheader("📄 Full Event Log")
        st.text_area(
            "Events",
            "\n".join([f"{ts:.1f}s | {who} | {evt}" for ts, who, evt in event_log]),
            height=220
        )

else:
    st.info("👆 Upload a video to start monitoring.")
    st.markdown("""
**What this app does**
- Uses a pre-trained AI model to detect **people**
- Uses AI tracking to maintain **stable IDs**
- Triggers **Intrusion** on **entry** into the restricted zone
- Triggers **Loitering** when a person remains inside the zone longer than the threshold
""")
