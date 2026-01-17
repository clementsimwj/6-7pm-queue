from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

app = FastAPI()

# YOLOv8 pretrained model
model = YOLO("yolo26m-pose.pt")
cap = cv2.VideoCapture(0)

#Paramters
MIN_QUEUE_HEIGHT_RATIO = 0.19 # Filters Distant People not in Queue


# Base route
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Upload Photo</h2>
            <form action="/count_debug" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <br><br>
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/count_debug")
async def count_queue_debug(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_image.size
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(pil_image, conf=0.25)
    all_keypoints = []
    
    for person in results[0].keypoints.xy:
        all_keypoints.append(np.array(person))
    if len(all_keypoints) == 0:
        return {"people_detected": 0}
    all_keypoints = np.vstack(all_keypoints)

    queue_count = 0
    
    boxes = results[0].boxes.xyxy
    keypoints_all = results[0].keypoints.xy

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        bbox_height = y2 - y1
        relative_height = bbox_height / height

        # Check if this person is a queue member
        in_queue = relative_height >= MIN_QUEUE_HEIGHT_RATIO
        
        angle = estimate_head_rotation_from_ear_width(keypoints_all[i].cpu().numpy())
        # Head keypoints indices
        head_kp_indices = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear

        # Extract head points for this person
        head_pts = keypoints_all[i].cpu().numpy()[head_kp_indices]

        # Remove missing points (where x=0 and y=0)
        valid_pts = head_pts[~np.any(head_pts == 0, axis=1)]

        if len(valid_pts) > 0:
            # Get tight bounding box around head
            x_min = int(np.min(valid_pts[:, 0]))
            y_min = int(np.min(valid_pts[:, 1]))
            x_max = int(np.max(valid_pts[:, 0]))
            y_max = int(np.max(valid_pts[:, 1]))

            # Optional: expand box slightly for better visualization
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.3)
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(width, x_max + pad_x)
            y_max = min(height, y_max + pad_y)

            # Draw head box (blue)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw head rotation angle above head
            if angle is not None:
                cv2.putText(
                    image,
                    f"Head: {angle}°",
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        
        # Draw bounding boxes
        color = (0, 255, 0) if in_queue else (0, 0, 255)  # Green = queue, Red = ignored
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if in_queue:
            queue_count += 1
    # Convert back to PIL and then to bytes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(image)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG")
    buf.seek(0)

    print(f"Persons in queue: {queue_count}")
    return StreamingResponse(buf, media_type="image/jpeg")


def count_heads_feet_in_box(keypoints, box, eps=20):
    """
    keypoints: Nx2 array of all detected keypoints in image
    box: [x1, y1, x2, y2] bounding box
    eps: distance threshold for clustering
    Returns estimated number of people in this box
    """
    x1, y1, x2, y2 = box
    # Filter keypoints inside the box
    kp_in_box = keypoints[(keypoints[:,0] >= x1) & (keypoints[:,0] <= x2) &
                           (keypoints[:,1] >= y1) & (keypoints[:,1] <= y2)]
    if len(kp_in_box) == 0:
        return 0

    # Separate head vs feet keypoints
    head_indices = [0,1,2,3,4]   # nose, eyes, ears
    feet_indices = [15,16]        # left/right ankle

    # Extract head and foot positions
    head_pts = kp_in_box[head_indices, :] if len(kp_in_box) > max(head_indices) else np.empty((0,2))
    feet_pts = kp_in_box[feet_indices, :] if len(kp_in_box) > max(feet_indices) else np.empty((0,2))

    # Cluster heads and feet separately
    def cluster_count(points):
        if len(points) == 0:
            return 0
        clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
        return len(set(clustering.labels_))
    
    head_count = cluster_count(head_pts)
    feet_count = cluster_count(feet_pts)

    # Return max of head/feet counts
    return max(head_count, feet_count)

def face_width(keypoints):
    fw = face_width_from_ear(keypoints)
    if fw is None or fw < 2:
        return 180

def face_width_from_ear(keypoints):
    """
    keypoints: (17,2) array of keypoints for a single person
    Tracks nose, eyes, and ears to estimate head rotation.
    Returns angle in degrees:
    0 = facing camera (both eyes visible, symmetric)
    90 = side (one ear visible, one eye hidden)
    180 = facing away (back of head)
    """
    # YOLO pose keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    
    face_points = [nose, left_eye, right_eye]
    valid_pts = np.array([pt for pt in face_points if not np.any(pt == 0)])
    
    if len(valid_pts) == 0:
        return None
    
    left_ear_visible = not np.any(left_ear == 0)
    right_ear_visible = not np.any(right_ear == 0)
    
    if left_ear_visible:
        # Distance from left ear to farthest facial point
        distances = valid_pts[:,0] - left_ear[0]
        width = np.max(distances)
        return width
    elif right_ear_visible:
        # Distance from right ear to farthest facial point
        distances = right_ear[0] - valid_pts[:,0]
        width = np.max(distances)
        return width
    else:
        # No ears visible, fallback to eye distance
        if not np.any(left_eye == 0) and not np.any(right_eye == 0):
            return abs(right_eye[0] - left_eye[0])
        return None

def estimate_head_rotation_from_ear_width(keypoints):
    """
    Estimates head rotation using visible ear to farthest facial point.
    Returns:
        angle (int): 0 = facing camera, 90 = facing left, 180 = facing back, 270 = facing right
    """

    # YOLO pose keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # Check shoulder keypoints
    if np.any(left_shoulder == 0) or np.any(right_shoulder == 0):
        return None

    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    if shoulder_width < 5:
        return None  # too small, ignore

    # Collect valid facial points (nose + eyes)
    face_points = [nose, left_eye, right_eye]
    valid_pts = np.array([pt for pt in face_points if not np.any(pt == 0)])
    if len(valid_pts) == 0:
        return None

    # Check visible ears
    left_ear_visible = not np.any(left_ear == 0)
    right_ear_visible = not np.any(right_ear == 0)

    # Compute face width based on visible ear to determine turn angle
    if left_ear_visible:
        # Left ear visible = head turned right (away from left)
        distances = valid_pts[:,0] - left_ear[0]
        fw_ratio = np.clip(np.max(distances) / shoulder_width, 0.0, 1.0)
        # Map: full face (fw_ratio=0) = 0°, half face (fw_ratio=0.5) = 270°, no face (fw_ratio=1) = 180°
        angle = fw_ratio * 270
    elif right_ear_visible:
        # Right ear visible = head turned left (away from right)
        distances = right_ear[0] - valid_pts[:,0]
        fw_ratio = np.clip(np.max(distances) / shoulder_width, 0.0, 1.0)
        # Map: full face (fw_ratio=0) = 0°, half face (fw_ratio=0.5) = 90°, no face (fw_ratio=1) = 180°
        angle = 360 - fw_ratio * 270
    else:
        # Both ears hidden = facing directly forward
        if not np.any(left_eye == 0) and not np.any(right_eye == 0):
            eye_distance = abs(right_eye[0] - left_eye[0])
            fw_ratio = np.clip(eye_distance / shoulder_width, 0.0, 1.0)
            # Eyes fully visible and wide apart = facing camera (0°)
            angle = 0
        else:
            return 180  # no face visible → back-facing

    # Normalize to 0-360 range
    angle = angle % 360
    return int(angle)
