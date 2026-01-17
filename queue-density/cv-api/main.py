from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math
from pathlib import Path

app = FastAPI()

# YOLOv8 pretrained model
model = YOLO("yolov8m-pose.pt")

#Paramters
MIN_QUEUE_HEIGHT_RATIO = 0.19 # Filters Distant People not in Queue


# Base route
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Queue Analysis System</h2>
            <h3>Upload Photo for Queue Analysis (with feet direction)</h3>
            <form action="/analyze_queue" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <br><br>
                <input type="submit" value="Analyze Queue">
            </form>
            <hr>
            <h3>Upload Photo for Basic Count (debug view)</h3>
            <form action="/count_debug" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <br><br>
                <input type="submit" value="Count Debug">
            </form>
            <hr>
            <p><a href="/test_assets">Test all sample images</a></p>
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

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        bbox_height = y2 - y1
        relative_height = bbox_height / height

        # Check if this person is a queue member
        in_queue = True
        if relative_height < MIN_QUEUE_HEIGHT_RATIO:
            in_queue = False

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


def calculate_foot_direction_with_toes(left_ankle, right_ankle, left_toe, right_toe,
                                       left_knee, right_knee, left_hip=None, right_hip=None, 
                                       left_shoulder=None, right_shoulder=None):
    """
    Calculate where the feet/toes are pointing using detected/estimated toe positions.
    Returns angle in degrees (0-360) where 0 is right, 90 is down, 180 is left, 270 is up
    """
    
    toe_directions = []
    
    # PRIORITY 1: Use ankle-to-toe vectors (most direct)
    if left_ankle is not None and left_toe is not None:
        left_direction = left_toe - left_ankle
        toe_directions.append(left_direction)
    
    if right_ankle is not None and right_toe is not None:
        right_direction = right_toe - right_ankle
        toe_directions.append(right_direction)
    
    # If we have toe directions, average them
    if len(toe_directions) > 0:
        facing_vector = sum(toe_directions) / len(toe_directions)
    
    # FALLBACK: No toe data, use body orientation
    elif left_ankle is not None and right_ankle is not None:
        # Use perpendicular to ankle line
        ankle_line = right_ankle - left_ankle
        perp1 = np.array([-ankle_line[1], ankle_line[0]])
        perp2 = np.array([ankle_line[1], -ankle_line[0]])
        
        # Use SHOULDERS to determine facing direction (most reliable for front/back orientation)
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            
            # Vector from shoulders to ankles
            shoulder_to_ankle = ankle_center - shoulder_center
            
            # The facing direction should be perpendicular to ankle line
            # AND should point away from the body center
            # Choose the perpendicular that points away from shoulders
            dot1 = np.dot(perp1, shoulder_to_ankle)
            dot2 = np.dot(perp2, shoulder_to_ankle)
            
            # Pick the one that aligns with body direction (shoulders -> feet)
            facing_vector = perp1 if dot1 > dot2 else perp2
        
        # Use knee/hip to determine which perpendicular
        elif left_knee is not None and right_knee is not None:
            knee_center = (left_knee + right_knee) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            body_dir = ankle_center - knee_center
            facing_vector = perp1 if np.dot(perp1, body_dir) > np.dot(perp2, body_dir) else perp2
        else:
            facing_vector = perp1
    
    else:
        return None
    
    # Calculate angle
    angle = math.atan2(facing_vector[1], facing_vector[0])
    angle_deg = math.degrees(angle)
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def calculate_foot_direction(left_ankle, right_ankle, left_knee, right_knee, left_hip=None, right_hip=None, left_shoulder=None, right_shoulder=None):
    """
    Calculate where the feet/toes are pointing using body orientation.
    Since YOLO doesn't detect toes, we use the body's forward direction.
    Returns angle in degrees (0-360) where 0 is right, 90 is down, 180 is left, 270 is up
    """
    
    facing_vector = None
    
    # PRIORITY 1: Use full body orientation (shoulders/hips to feet)
    # This shows the direction the torso is facing
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Get foot/ankle center
        if left_ankle is not None and right_ankle is not None:
            ankle_center = (left_ankle + right_ankle) / 2
        elif left_ankle is not None:
            ankle_center = left_ankle
        elif right_ankle is not None:
            ankle_center = right_ankle
        else:
            ankle_center = None
        
        if ankle_center is not None:
            # The direction from shoulders to ankles shows body orientation
            # But we want perpendicular to this (where they're facing, not where their body points)
            shoulder_to_ankle = ankle_center - shoulder_center
            # Perpendicular gives the facing direction
            perp1 = np.array([-shoulder_to_ankle[1], shoulder_to_ankle[0]])
            perp2 = np.array([shoulder_to_ankle[1], -shoulder_to_ankle[0]])
            
            # Use hip-shoulder orientation to pick the correct perpendicular
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_line = right_shoulder - left_shoulder
                # Pick perpendicular that points in same general direction as shoulder line
                facing_vector = perp1 if abs(np.dot(perp1, shoulder_line)) > abs(np.dot(perp2, shoulder_line)) else perp2
            else:
                facing_vector = perp1
    
    # PRIORITY 2: Use hip to ankle direction
    elif (left_hip is not None or right_hip is not None) and (left_ankle is not None or right_ankle is not None):
        if left_hip is not None and right_hip is not None:
            hip_center = (left_hip + right_hip) / 2
        elif left_hip is not None:
            hip_center = left_hip
        else:
            hip_center = right_hip
        
        if left_ankle is not None and right_ankle is not None:
            ankle_center = (left_ankle + right_ankle) / 2
        elif left_ankle is not None:
            ankle_center = left_ankle
        else:
            ankle_center = right_ankle
        
        # Hip to ankle shows body direction, perpendicular is facing
        hip_to_ankle = ankle_center - hip_center
        facing_vector = np.array([-hip_to_ankle[1], hip_to_ankle[0]])
    
    # PRIORITY 3: Use ankle line perpendicular with knee reference
    elif left_ankle is not None and right_ankle is not None:
        ankle_line = right_ankle - left_ankle
        perp1 = np.array([-ankle_line[1], ankle_line[0]])
        perp2 = np.array([ankle_line[1], -ankle_line[0]])
        
        # Use knee positions to determine which perpendicular
        if left_knee is not None and right_knee is not None:
            knee_center = (left_knee + right_knee) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            body_dir = ankle_center - knee_center
            facing_vector = perp1 if np.dot(perp1, body_dir) > np.dot(perp2, body_dir) else perp2
        else:
            facing_vector = perp1
    
    else:
        return None
    
    # Calculate angle
    angle = math.atan2(facing_vector[1], facing_vector[0])
    angle_deg = math.degrees(angle)
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def calculate_foot_direction_geometric(left_ankle, right_ankle, left_knee, right_knee, 
                                       left_hip=None, right_hip=None, left_shoulder=None, right_shoulder=None):
    """
    Calculate foot direction using geometric relationships between body parts.
    Key insight: shoulder orientation tells us body rotation, which determines foot direction.
    Returns angle in degrees (0-360) where 0 is right, 90 is down, 180 is left, 270 is up
    """
    
    # Need at least both ankles
    if left_ankle is None or right_ankle is None:
        return None
    
    # The line between ankles represents the width of the stance
    ankle_line = right_ankle - left_ankle
    
    # The two possible perpendicular directions (forward and backward)
    perp_forward = np.array([-ankle_line[1], ankle_line[0]])
    perp_backward = np.array([ankle_line[1], -ankle_line[0]])
    
    # Use shoulder line to determine body facing direction
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_line = right_shoulder - left_shoulder
        
        # If shoulders and ankles are roughly parallel, person is facing perpendicular to those lines
        # The shoulder line and ankle line should point in similar directions when viewed from above
        # Cross product tells us if we need forward or backward perpendicular
        
        # Use dot product: if shoulder line and ankle line point in same direction,
        # then the person is facing in the perpendicular direction
        similarity = np.dot(ankle_line / (np.linalg.norm(ankle_line) + 1e-6), 
                          shoulder_line / (np.linalg.norm(shoulder_line) + 1e-6))
        
        # If lines are parallel (dot product close to 1 or -1), use perpendicular
        # Choose based on which side the shoulders are relative to ankle center
        ankle_center = (left_ankle + right_ankle) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Vector from ankle to shoulder (upward in standing pose)
        ankle_to_shoulder = shoulder_center - ankle_center
        
        # The facing direction is perpendicular to ankle line,
        # Choose the one that when combined with ankle_to_shoulder doesn't point too far back
        # Essentially: if shoulders are behind you relative to facing direction, pick the other one
        
        dot_forward = np.dot(perp_forward, ankle_to_shoulder)
        dot_backward = np.dot(perp_backward, ankle_to_shoulder)
        
        # Pick the perpendicular that is more perpendicular to the ankle-to-shoulder vector
        # (i.e., doesn't point up or down too much, points more horizontally)
        facing_vector = perp_forward if abs(dot_forward) < abs(dot_backward) else perp_backward
        
    # Fallback: use knee positions
    elif left_knee is not None and right_knee is not None:
        knee_center = (left_knee + right_knee) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        
        # Knees to ankles vector
        knee_to_ankle = ankle_center - knee_center
        
        # Choose perpendicular more orthogonal to knee-ankle direction
        dot_forward = np.dot(perp_forward, knee_to_ankle)
        dot_backward = np.dot(perp_backward, knee_to_ankle)
        
        facing_vector = perp_forward if abs(dot_forward) < abs(dot_backward) else perp_backward
    else:
        # Default
        facing_vector = perp_forward
    
    # Calculate angle
    angle = math.atan2(facing_vector[1], facing_vector[0])
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def get_keypoint(keypoints, index, confidence_threshold=0.3):
    """
    Extract a keypoint if it exists and has sufficient confidence
    Returns [x, y] or None
    """
    if index < len(keypoints):
        kp = keypoints[index]
        # Check if keypoint is valid (non-zero)
        if kp[0] > 0 or kp[1] > 0:
            return np.array([kp[0], kp[1]])
    return None


def calculate_queue_alignment_score(direction_degrees):
    """
    Convert direction angle to queue alignment score.
    180° (left) = 1.0 (perfectly aligned with queue/stall)
    0° (right) = 0.0 (facing away from queue)
    
    Uses cosine relationship: score = (1 - cos(angle)) / 2
    """
    if direction_degrees is None:
        return None
    
    angle_radians = math.radians(direction_degrees)
    score = (1 - math.cos(angle_radians)) / 2
    return score


@app.post("/analyze_queue")
async def analyze_queue(file: UploadFile = File(...)):
    """
    Analyze queue using feet detection and direction
    Returns: count, average direction, and annotated image
    """
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_image.size
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO pose detection
    results = model(pil_image, conf=0.25)
    
    if len(results[0].keypoints.xy) == 0:
        return JSONResponse({
            "people_count": 0,
            "queue_length": 0,
            "average_direction": None,
            "message": "No people detected"
        })

    queue_members = []
    boxes = results[0].boxes.xyxy
    keypoints_data = results[0].keypoints.xy

    # COCO keypoint indices:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    for idx, (box, keypoints) in enumerate(zip(boxes, keypoints_data)):
        x1, y1, x2, y2 = map(int, box.tolist())
        bbox_height = y2 - y1
        relative_height = bbox_height / height

        # Filter out distant people
        if relative_height < MIN_QUEUE_HEIGHT_RATIO:
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            continue

        # Extract keypoints
        left_ankle = get_keypoint(keypoints, 15)
        right_ankle = get_keypoint(keypoints, 16)
        left_knee = get_keypoint(keypoints, 13)
        right_knee = get_keypoint(keypoints, 14)
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)

        # Check if at least both ankles are detected
        if left_ankle is None or right_ankle is None:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "Need both feet", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Calculate direction using geometric method
        direction = calculate_foot_direction_geometric(left_ankle, right_ankle, left_knee, right_knee, 
                                                       left_hip, right_hip, left_shoulder, right_shoulder)
        
        # Draw keypoints for debugging
        if left_ankle is not None:
            cv2.circle(image, tuple(left_ankle.astype(int)), 5, (255, 0, 0), -1)  # Blue
        if right_ankle is not None:
            cv2.circle(image, tuple(right_ankle.astype(int)), 5, (0, 255, 0), -1)  # Green
        if left_knee is not None:
            cv2.circle(image, tuple(left_knee.astype(int)), 5, (255, 128, 0), -1)  # Orange
        if right_knee is not None:
            cv2.circle(image, tuple(right_knee.astype(int)), 5, (0, 128, 255), -1)  # Light blue
        
        # Draw knee-to-ankle lines
        if left_knee is not None and left_ankle is not None:
            cv2.line(image, tuple(left_knee.astype(int)), tuple(left_ankle.astype(int)), (255, 0, 255), 2)
        if right_knee is not None and right_ankle is not None:
            cv2.line(image, tuple(right_knee.astype(int)), tuple(right_ankle.astype(int)), (255, 0, 255), 2)

        # Calculate center point for direction arrow
        if left_ankle is not None and right_ankle is not None:
            center = ((left_ankle + right_ankle) / 2).astype(int)
        elif left_ankle is not None:
            center = left_ankle.astype(int)
        else:
            center = right_ankle.astype(int)

        # Draw direction arrow
        if direction is not None:
            # Calculate queue alignment score (1.0 = facing left/stall, 0.0 = facing right/away)
            alignment_score = calculate_queue_alignment_score(direction)
            
            arrow_length = 40
            end_x = int(center[0] + arrow_length * math.cos(math.radians(direction)))
            end_y = int(center[1] + arrow_length * math.sin(math.radians(direction)))
            cv2.arrowedLine(image, tuple(center), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
            
            # Draw bounding box and alignment score
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{alignment_score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            queue_members.append({
                "id": idx,
                "direction": direction,
                "alignment_score": alignment_score,
                "position": center.tolist(),
                "bbox": [x1, y1, x2, y2]
            })

    # Calculate statistics
    queue_length = len(queue_members)
    if queue_length > 0:
        directions = [m["direction"] for m in queue_members]
        avg_direction = sum(directions) / len(directions)
    else:
        avg_direction = None

    # Convert back to PIL and then to bytes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(image)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg", 
                           headers={
                               "X-Queue-Length": str(queue_length),
                               "X-Average-Direction": str(avg_direction) if avg_direction else "null"
                           })


@app.get("/test_assets")
async def test_assets():
    """
    Test endpoint to analyze all images in Assets/SamplePics
    """
    assets_path = Path("Assets/SamplePics")
    if not assets_path.exists():
        return {"error": "Assets/SamplePics directory not found"}
    
    results = []
    for img_file in assets_path.glob("*.jpg"):
        with open(img_file, "rb") as f:
            image_bytes = f.read()
        
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size
        
        # Run YOLO pose detection
        detection_results = model(pil_image, conf=0.25)
        
        if len(detection_results[0].keypoints.xy) == 0:
            results.append({
                "file": img_file.name,
                "queue_length": 0,
                "people_detected": 0
            })
            continue
        
        queue_count = 0
        boxes = detection_results[0].boxes.xyxy
        keypoints_data = detection_results[0].keypoints.xy
        
        for box, keypoints in zip(boxes, keypoints_data):
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox_height = y2 - y1
            relative_height = bbox_height / height
            
            if relative_height < MIN_QUEUE_HEIGHT_RATIO:
                continue
            
            # Check for feet
            left_ankle = get_keypoint(keypoints, 15)
            right_ankle = get_keypoint(keypoints, 16)
            
            if left_ankle is not None or right_ankle is not None:
                queue_count += 1
        
        results.append({
            "file": img_file.name,
            "queue_length": queue_count,
            "people_detected": len(boxes)
        })
    
    return {"results": results}
