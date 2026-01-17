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
model = YOLO("yolov8m-pose.pt")

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
