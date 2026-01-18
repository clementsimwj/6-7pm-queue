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
CONSECUTIVE_PERSON_DISTANCE_THRESHOLD = 260  # Distance threshold between consecutive persons (in pixels)


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
    
    # Run YOLO detection
    results = model(pil_image, conf=0.25)
    all_keypoints = []
    
    for person in results[0].keypoints.xy:
        all_keypoints.append(np.array(person))
    if len(all_keypoints) == 0:
        return {"people_detected": 0}
    all_keypoints = np.vstack(all_keypoints)

    queue_count = 0
    valid_persons = []  # Store valid person boxes: [x1, y1, x2, y2, person_index]
    person_in_queue_status = {}  # Track in_queue status by original person index
    
    boxes = results[0].boxes.xyxy
    keypoints_all = results[0].keypoints.xy

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        bbox_height = y2 - y1
        relative_height = bbox_height / height

        keypoints = keypoints_all[i].cpu().numpy()
        angle = estimate_head_rotation_from_keypoints(keypoints)[0]
        
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
            
            # Check if head is in bottom region
            head_box = [x_min, y_min, x_max, y_max]
            head_in_bottom_region = is_in_queue_region(head_box, queue_region_bottom, threshold=0.5)

        angle_ok = 0 <= angle <= 10 if angle is not None else False
        # Check if person is in either queue region (middle or bottom)
        in_middle_region = is_in_queue_region(box.tolist(), queue_region_middle, QUEUE_REGION_THRESHOLD)
        in_bottom_region = is_in_queue_region(box.tolist(), queue_region_bottom, QUEUE_REGION_THRESHOLD)
        
        # ==================== QUEUE DETECTION LOGIC ====================

        
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
        
        if in_queue:
            queue_count += 1
            # Store valid person's bounding box for distance calculation and track index
            valid_persons.append({'box': [x1, y1, x2, y2], 'original_index': i})
            person_in_queue_status[i] = True
    
    # ==================== CALCULATE DISTANCES BETWEEN VALID PERSONS (logic only) ====================
    # Starting point: left-most corner of middle region
    region_start_point = (queue_x_start, queue_y_start)
    
    if len(valid_persons) > 0:
        # Calculate left-hand corners and distances for each person
        persons_with_distances = []
        for person_data in valid_persons:
            person_box = person_data['box']
            original_index = person_data['original_index']
            x1, y1, x2, y2 = person_box
            left_corner = (x1, y1)  # Top left corner
            
            # Calculate distance from region start to this person's leftmost corner
            distance_from_start = math.sqrt((region_start_point[0] - left_corner[0])**2 + (region_start_point[1] - left_corner[1])**2)
            
            # Store person data: [person_index, box, left_corner, distance]
            persons_with_distances.append({
                'index': original_index,
                'box': person_box,
                'left_corner': left_corner,
                'distance': distance_from_start
            })
        
        # Sort persons by distance
        sorted_persons = sorted(persons_with_distances, key=lambda p: p['distance'])
        
        # Identify valid queue members based on consecutive distance threshold
        queue_members = []
        queue_broken = False
        excluded_indices = []  # Track indices of excluded persons
        
        # Check distance from region start to first person
        first_person = sorted_persons[0]
        distance_to_first = math.sqrt((region_start_point[0] - first_person['left_corner'][0])**2 + 
                                      (region_start_point[1] - first_person['left_corner'][1])**2)
        
        if distance_to_first <= CONSECUTIVE_PERSON_DISTANCE_THRESHOLD:
            queue_members.append(first_person)
        else:
            queue_broken = True
            excluded_indices.append(first_person['index'])
        
        # Check consecutive distances
        for idx in range(1, len(sorted_persons)):
            if queue_broken:
                excluded_indices.append(sorted_persons[idx]['index'])
                continue
            
            prev_person = sorted_persons[idx - 1]
            curr_person = sorted_persons[idx]
            
            # Calculate distance between consecutive persons
            consecutive_distance = math.sqrt((prev_person['left_corner'][0] - curr_person['left_corner'][0])**2 + 
                                           (prev_person['left_corner'][1] - curr_person['left_corner'][1])**2)
            
            # Check if distance exceeds threshold
            if consecutive_distance > CONSECUTIVE_PERSON_DISTANCE_THRESHOLD:
                queue_broken = True
                excluded_indices.append(curr_person['index'])
            else:
                queue_members.append(curr_person)
        
        # Update person_in_queue_status to invalidate excluded persons
        for excluded_idx in excluded_indices:
            person_in_queue_status[excluded_idx] = False
        
        # Print sorted persons for debugging
        print(f"Threshold for queue break: {CONSECUTIVE_PERSON_DISTANCE_THRESHOLD}px")
        print("Sorted persons by distance from region start:")
        for idx, person_data in enumerate(sorted_persons):
            status = "IN QUEUE" if person_data in queue_members else "EXCLUDED (queue broken)"
            print(f"  Person {idx + 1}: Distance from start = {person_data['distance']:.1f}px, Status: {status}")
        
        print(f"\nQueue members in valid queue: {len(queue_members)}")
        
        # Print distance from start to first person
        print(f"\nDistance from region start to first person: {distance_to_first:.1f}px - ", end="")
        if distance_to_first <= CONSECUTIVE_PERSON_DISTANCE_THRESHOLD:
            print("OK (within threshold)")
        else:
            print("EXCEEDS THRESHOLD - QUEUE BROKEN FROM START")
        
        print("Consecutive distances:")
        for idx in range(1, len(sorted_persons)):
            prev_person = sorted_persons[idx - 1]
            curr_person = sorted_persons[idx]
            distance = math.sqrt((prev_person['left_corner'][0] - curr_person['left_corner'][0])**2 + 
                               (prev_person['left_corner'][1] - curr_person['left_corner'][1])**2)
            status = "OK" if distance <= CONSECUTIVE_PERSON_DISTANCE_THRESHOLD else "EXCEEDS THRESHOLD - QUEUE BROKEN"
            print(f"  Person {idx} to Person {idx + 1}: {distance:.1f}px - {status}")
    
    # After processing all persons and determining queue breaks, redraw boxes based on final status
    for original_index, in_queue_status in person_in_queue_status.items():
        # Find this person in boxes and redraw with appropriate color
        box = results[0].boxes.xyxy[original_index]
        x1, y1, x2, y2 = map(int, box.tolist())
        
        if in_queue_status:
            # Green for persons in final valid queue
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            # Red for persons excluded due to queue break
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # Convert back to PIL and then to bytes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(image)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG")
    buf.seek(0)

    # Calculate actual queue count based on final person_in_queue_status
    final_queue_count = sum(1 for status in person_in_queue_status.values() if status)
    
    print(f"Persons initially detected in queue region: {queue_count}")
    print(f"Persons in final valid queue (after distance threshold): {final_queue_count}")
    print(f"Persons excluded due to queue break: {queue_count - final_queue_count}")
    return StreamingResponse(buf, media_type="image/jpeg", headers={
        "X-Queue-Count": str(final_queue_count)
    })


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

def estimate_head_rotation_from_keypoints(keypoints):
    """
    Estimates head yaw rotation using YOLO pose keypoints.
    
    Args:
        keypoints: Array of shape (N, 2) or (N, 3) where N >= 7
                   [0]=nose, [1]=left_eye, [2]=right_eye, 
                   [3]=left_ear, [4]=right_ear,
                   [5]=left_shoulder, [6]=right_shoulder
    
    Returns:
        angle (int): Yaw angle in degrees
                     0° = facing camera
                     90° = facing left  
                     -90° (or 270°) = facing right
                     ±180° = facing away
        confidence (float): 0.0 to 1.0 indicating estimation confidence
    """
    
    # Extract keypoints
    nose = keypoints[0][:2]
    left_eye = keypoints[1][:2]
    right_eye = keypoints[2][:2]
    left_ear = keypoints[3][:2]
    right_ear = keypoints[4][:2]
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    
    # Check validity (assuming [0,0] or very low confidence means invalid)
    def is_valid(pt):
        return not (pt[0] < 1 and pt[1] < 1)
    
    # -----------------------------
    # Method 1: Eye-Nose Triangle (most reliable for frontal views)
    # -----------------------------
    if is_valid(nose) and is_valid(left_eye) and is_valid(right_eye):
        # Calculate face center from eyes
        eye_center = (left_eye + right_eye) / 2
        
        # Vector from eye center to nose
        eye_to_nose = nose - eye_center
        
        # Vector along eye line (left to right)
        eye_line = right_eye - left_eye
        eye_distance = np.linalg.norm(eye_line)
        
        if eye_distance > 5:  # Minimum threshold
            # Normalize eye line
            eye_line_norm = eye_line / eye_distance
            
            # Project nose displacement onto eye line (lateral offset)
            lateral_offset = np.dot(eye_to_nose, eye_line_norm)
            
            # Asymmetry ratio: how far nose is from eye midpoint
            asymmetry = lateral_offset / eye_distance
            
            # This gives us a good yaw estimate for ±60° range
            # asymmetry: -0.5 to +0.5 maps roughly to -60° to +60°
            yaw_from_eyes = np.clip(asymmetry * 120, -75, 75)
        else:
            yaw_from_eyes = None
    else:
        yaw_from_eyes = None
    
    # -----------------------------
    # Method 2: Ear Visibility Analysis
    # -----------------------------
    left_ear_vis = is_valid(left_ear)
    right_ear_vis = is_valid(right_ear)
    
    yaw_from_ears = None
    ear_confidence = 0.0
    
    if left_ear_vis or right_ear_vis:
        # Calculate face center
        face_points = [p for p, v in [(nose, is_valid(nose)), 
                                       (left_eye, is_valid(left_eye)), 
                                       (right_eye, is_valid(right_eye))] if v]
        
        if len(face_points) > 0:
            face_center = np.mean(face_points, axis=0)
            
            # Reference width (shoulder or eye distance)
            if is_valid(left_shoulder) and is_valid(right_shoulder):
                ref_width = np.linalg.norm(right_shoulder - left_shoulder)
            elif is_valid(left_eye) and is_valid(right_eye):
                ref_width = np.linalg.norm(right_eye - left_eye) * 2.5
            else:
                ref_width = 100  # fallback
            
            if left_ear_vis and not right_ear_vis:
                # Head turned to the right (left ear visible)
                ear_to_face = np.linalg.norm(left_ear - face_center)
                ratio = np.clip(ear_to_face / ref_width, 0.0, 1.5)
                # Maps to 45° to 135° range
                yaw_from_ears = 45 + ratio * 60
                ear_confidence = 0.7
                
            elif right_ear_vis and not left_ear_vis:
                # Head turned to the left (right ear visible)
                ear_to_face = np.linalg.norm(right_ear - face_center)
                ratio = np.clip(ear_to_face / ref_width, 0.0, 1.5)
                # Maps to -45° to -135° range
                yaw_from_ears = -45 - ratio * 60
                ear_confidence = 0.7
                
            elif left_ear_vis and right_ear_vis:
                # Both ears visible
                left_dist = np.linalg.norm(left_ear - face_center)
                right_dist = np.linalg.norm(right_ear - face_center)
                
                # Check if one ear is much farther (indicates rotation)
                dist_ratio = abs(left_dist - right_dist) / max(left_dist, right_dist, 1)
                
                if dist_ratio < 0.3:
                    # Similar distances - likely frontal or back view
                    if is_valid(nose):
                        yaw_from_ears = 0  # frontal
                        ear_confidence = 0.5
                    else:
                        yaw_from_ears = 180  # back view
                        ear_confidence = 0.4
                else:
                    # Asymmetric - use ear distances
                    if left_dist > right_dist:
                        yaw_from_ears = 30  # slight right turn
                    else:
                        yaw_from_ears = -30  # slight left turn
                    ear_confidence = 0.4
    
    # -----------------------------
    # Combine Methods
    # -----------------------------
    if yaw_from_eyes is not None and yaw_from_ears is not None:
        # Weighted average based on angle magnitude
        # Trust eye method more for frontal views
        eye_weight = max(0.3, 1.0 - abs(yaw_from_eyes) / 90)
        ear_weight = ear_confidence
        
        total_weight = eye_weight + ear_weight
        final_yaw = (yaw_from_eyes * eye_weight + yaw_from_ears * ear_weight) / total_weight
        confidence = min(0.9, (eye_weight + ear_weight) / 2)
        
    elif yaw_from_eyes is not None:
        final_yaw = yaw_from_eyes
        confidence = 0.7
        
    elif yaw_from_ears is not None:
        final_yaw = yaw_from_ears
        confidence = ear_confidence
        
    else:
        # Fallback: no reliable features detected
        if is_valid(nose):
            final_yaw = 0
        else:
            final_yaw = 180
        confidence = 0.1
    
    # Convert to 0-360 range if needed
    angle = int(final_yaw % 360)
    
    return angle, round(confidence, 2)


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