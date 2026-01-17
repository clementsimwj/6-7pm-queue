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
model = YOLO("yolo26m-pose.pt")
cap = cv2.VideoCapture(0)

#Paramters
MIN_QUEUE_HEIGHT_RATIO = 0.20 # Filters Distant People not in Queue
QUEUE_REGION = None  # Will be set based on image - define your queue region here
QUEUE_REGION_THRESHOLD = 0.6  # 60% of person must be in region


# Base route
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h3>Upload Photo for Basic Count (debug view)</h3>
            <form action="/count_debug" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <br><br>
                <input type="submit" value="Count Debug">
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

    # Get queue regions - customize here
    # Middle region: Horizontal rectangle spanning entire width (offset 20% left), 40% height starting from middle
    queue_x_start = int(width * 0.15)
    queue_y_start = int(height / 2)
    queue_y_end = int(height / 2 + height * 0.4)
    queue_region_middle = (queue_x_start, queue_y_start, width, queue_y_end)
    
    # Bottom region: Horizontal rectangle spanning entire width, 30% height starting from 70% down
    queue_y_bottom_start = int(height * 0.70)
    queue_y_bottom_end = height
    queue_region_bottom = (0, queue_y_bottom_start, width, queue_y_bottom_end)
    
    # Draw queue regions on image for visualization (cyan for middle, magenta for bottom)
    cv2.rectangle(image, (queue_region_middle[0], queue_region_middle[1]), (queue_region_middle[2], queue_region_middle[3]), (200, 200, 0), 2)
    cv2.rectangle(image, (queue_region_bottom[0], queue_region_bottom[1]), (queue_region_bottom[2], queue_region_bottom[3]), (255, 0, 255), 2)

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

        keypoints = keypoints_all[i].cpu().numpy()
        angle = estimate_head_rotation_from_ear_width(keypoints)
        
        # ==================== FEET DIRECTION ANALYSIS (from analyze_queue) ====================
        # Extract keypoints for feet direction
        left_ankle = get_keypoint(keypoints, 15)
        right_ankle = get_keypoint(keypoints, 16)
        left_knee = get_keypoint(keypoints, 13)
        right_knee = get_keypoint(keypoints, 14)
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)
        
        # Calculate foot direction using geometric method
        direction = None
        alignment_score = None
        if left_ankle is not None and right_ankle is not None:
            direction = calculate_foot_direction_geometric(left_ankle, right_ankle, left_knee, right_knee, 
                                                           left_hip, right_hip, left_shoulder, right_shoulder)
            if direction is not None:
                alignment_score = calculate_queue_alignment_score(direction)
        
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
            feet_center = ((left_ankle + right_ankle) / 2).astype(int)
        elif left_ankle is not None:
            feet_center = left_ankle.astype(int)
        elif right_ankle is not None:
            feet_center = right_ankle.astype(int)
        else:
            feet_center = None
        
        # Draw direction arrow
        if direction is not None and feet_center is not None:
            arrow_length = 40
            end_x = int(feet_center[0] + arrow_length * math.cos(math.radians(direction)))
            end_y = int(feet_center[1] + arrow_length * math.sin(math.radians(direction)))
            cv2.arrowedLine(image, tuple(feet_center), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
        
        # ==================== HEAD ANALYSIS (existing count_debug logic) ====================
        # Head keypoints indices
        head_kp_indices = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
        head_pts = keypoints[head_kp_indices]

        # Remove missing points (where x=0 and y=0)
        valid_pts = head_pts[~np.any(head_pts == 0, axis=1)]

        head_in_bottom_region = False
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
            
            # Check if head is in bottom region
            head_box = [x_min, y_min, x_max, y_max]
            head_in_bottom_region = is_in_queue_region(head_box, queue_region_bottom, threshold=0.5)

        angle_ok = 0 <= angle <= 10 if angle is not None else False
        # Check if person is in either queue region (middle or bottom)
        in_middle_region = is_in_queue_region(box.tolist(), queue_region_middle, QUEUE_REGION_THRESHOLD)
        in_bottom_region = is_in_queue_region(box.tolist(), queue_region_bottom, QUEUE_REGION_THRESHOLD)
        
        # Display region status above bounding box
        region_status = "Middle" if in_middle_region else ("Bottom" if in_bottom_region else "Outside")
        cv2.putText(
            image,
            f"Region: {region_status}",
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
        
        # ==================== DISPLAY ALIGNMENT SCORE ====================
        # Draw alignment score prominently (THIS IS WHAT YOU NEED TO SEE)
        if alignment_score is not None:
            cv2.putText(
                image,
                f"Align: {alignment_score:.2f}",
                (x1, y1 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # Cyan color to stand out
                2
            )
        
        # If head is detected in bottom region, exclude person from queue
        if head_in_bottom_region:
            in_queue = False
        else:
            # Original condition: head angle + region + height
            original_condition = angle_ok and in_middle_region and (relative_height >= MIN_QUEUE_HEIGHT_RATIO)
            
            # New condition: alignment score based detection
            alignment_condition = in_middle_region and (alignment_score is not None and 0.65 <= alignment_score <= 1.0)
            
            # Person is in queue if EITHER condition is met
            in_queue = original_condition or alignment_condition
        
        # Draw bounding boxes
        color = (0, 255, 0) if in_queue else (0, 0, 255)  # Green = in queue region, Red = outside
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


def get_queue_region(width, height, position="left", percentage=0.33):
    """
    Define the queue region dynamically based on image dimensions.
    
    Args:
        width: Image width
        height: Image height
        position: "left", "right", "center", or tuple (x1, y1, x2, y2) for custom region
        percentage: What percentage of width the region should cover (0.0 to 1.0)
    
    Returns:
        tuple: (x1, y1, x2, y2) - queue region coordinates
    """
    region_width = int(width * percentage)
    
    if isinstance(position, tuple):
        # Custom region provided
        return position
    elif position == "left":
        return (0, 0, region_width, height)
    elif position == "right":
        return (width - region_width, 0, width, height)
    elif position == "center":
        start_x = (width - region_width) // 2
        return (start_x, 0, start_x + region_width, height)
    else:
        return (0, 0, region_width, height)


def is_in_queue_region(bbox, region, threshold=0.6):
    """
    Check if a bounding box overlaps with the queue region by at least threshold percentage.
    
    Args:
        bbox: [x1, y1, x2, y2] - person's bounding box
        region: (x1, y1, x2, y2) - queue region box
        threshold: percentage of bbox that must be in region (0.0 to 1.0)
    
    Returns:
        bool: True if >= threshold% of bbox is within region
    """
    x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
    x1_region, y1_region, x2_region, y2_region = region
    
    # Calculate intersection box
    x1_inter = max(x1_bbox, x1_region)
    y1_inter = max(y1_bbox, y1_region)
    x2_inter = min(x2_bbox, x2_region)
    y2_inter = min(y2_bbox, y2_region)
    
    # If no intersection, return False
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return False
    
    # Calculate areas
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    bbox_area = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
    
    # Calculate overlap percentage
    overlap_ratio = intersection_area / bbox_area if bbox_area > 0 else 0
    
    return overlap_ratio >= threshold


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


    # Collect valid facial points (nose + eyes)
    face_points = [nose, left_eye, right_eye]
    valid_pts = np.array([pt for pt in face_points if not np.any(pt == 0)])
    if len(valid_pts) == 0:
        return None

    # --- Shoulder width estimation ---
    shoulder_width = None

    if not np.any(left_shoulder == 0) and not np.any(right_shoulder == 0):
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

    else:
        # Fallback: estimate shoulder width from head width
        if not np.any(left_eye == 0) and not np.any(right_eye == 0):
            eye_width = abs(right_eye[0] - left_eye[0])
            shoulder_width = eye_width * 2.2  # human proportion
        elif not np.any(left_ear == 0) or not np.any(right_ear == 0):
            face_x = valid_pts[:, 0]
            shoulder_width = (face_x.max() - face_x.min()) * 2.5

    if shoulder_width is None or shoulder_width < 5:
        shoulder_width = 50  # final fallback constant

    # --- Ear visibility ---
    left_ear_visible = not np.any(left_ear == 0)
    right_ear_visible = not np.any(right_ear == 0)

    # --- Angle estimation ---
    if left_ear_visible:
        distances = valid_pts[:, 0] - left_ear[0]
        fw_ratio = np.clip(np.max(distances) / shoulder_width, 0.0, 1.0)
        angle = fw_ratio * 270

    elif right_ear_visible:
        distances = right_ear[0] - valid_pts[:, 0]
        fw_ratio = np.clip(np.max(distances) / shoulder_width, 0.0, 1.0)
        angle = 360 - fw_ratio * 270

    else:
        # No ears → assume frontal
        angle = 0

    return int(angle % 360)


# ==================== NEW FUNCTIONS FROM FEET DIRECTION ANALYSIS ====================

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