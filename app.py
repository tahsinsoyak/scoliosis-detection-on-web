from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from ultralytics import YOLO
from cobb_angle import cobb_angle_cal
import json
import datetime
import uuid

app = Flask(__name__)

model_paths = {
    'l-bestmodel': 'static/models/l-best.pt',
    'm-bestmodel': 'static/models/m-best.pt',
    'n-bestmodel': 'static/models/n-best.pt',
    'x-bestmodel': 'static/models/x-best.pt'
}
def save_image_with_bounding_boxes(img, detections, file_path):
    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(file_path, img)

def save_image_with_middle_points(img, detections, file_path):
    vertebrae_points = []
    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        if conf > 0.5:  # Confidence threshold
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            vertebrae_points.append((center_x, center_y))
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw dots at the four corners of the bounding box
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(img, (x2, y1), 5, (255, 0, 0), -1)
            cv2.circle(img, (x1, y2), 5, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)
    
    vertebrae_points.sort(key=lambda point: point[1])
    for i in range(len(vertebrae_points) - 1):
        cv2.line(img, vertebrae_points[i], vertebrae_points[i+1], (255, 0, 0), 2)
    
    cv2.imwrite(file_path, img)
    return vertebrae_points

def detect_vertebrae(image_path, model_path):
    model = YOLO(model_path)  # Load the selected model
    img = cv2.imread(image_path)
    results = model(img)
    return results[0]  # Access the first element of the list

def prepare_landmark_xy(vertebrae_points):
    landmark_xy = []
    for point in vertebrae_points:
        landmark_xy.append(point[0])  # x-coordinate
    for point in vertebrae_points:
        landmark_xy.append(point[1])  # y-coordinate
    return landmark_xy

def calculate_cobb_angle(vertebrae_points, img, file_path):
    image_shape = img.shape
    landmark_xy = prepare_landmark_xy(vertebrae_points)
    
    cobb_angles_list, angles_with_pos, curve_type, midpoint_lines = cobb_angle_cal(landmark_xy, image_shape)
    
    output_path = file_path.replace('.jpg', '_cobb_angle.jpg')
    
    for line in midpoint_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        
        if x2 == x1:
            # Avoid division by zero if the line is vertical
            cv2.line(img, (x1, 0), (x1, image_shape[0] - 1), (0, 255, 0), 2)
        else:
            slope = (y2 - y1) / (x2 - x1)
            
            x1_extended = 0
            y1_extended = int(y1 - (x1 - x1_extended) * slope)
            x2_extended = image_shape[1] - 1
            y2_extended = int(y2 + (x2_extended - x2) * slope)
            
            cv2.line(img, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    return cobb_angles_list, output_path, curve_type

def load_results():
    try:
        if os.path.exists('results.json'):
            with open('results.json', 'r') as file:
                data = file.read()
                if data.strip():  # Check if the file is not empty
                    return json.loads(data)
                else:
                    return []  # Return an empty list if the file is empty
        else:
            return []  # Return an empty list if the file does not exist
    except json.JSONDecodeError:
        return []  # Return an empty list if the file contains invalid JSON

def save_result(result):
    results = load_results()
    result['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.append(result)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

def update_paths_in_results():
    with open('results.json', 'r') as file:
        results = json.load(file)

    for result in results:
        result['image_path'] = result['image_path'].replace("static/", "").replace("\\", "/")
        result['bounding_boxes_img_path'] = result['bounding_boxes_img_path'].replace("static/", "").replace("\\", "/")
        result['middle_points_img_path'] = result['middle_points_img_path'].replace("static/", "").replace("\\", "/")
        result['cobb_angle_img_path'] = result['cobb_angle_img_path'].replace("static/", "").replace("\\", "/")

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

def generate_unique_folder(upload_folder):
    folder_name = str(uuid.uuid4().hex[:8])  # Generate a unique folder name
    unique_folder = os.path.join(upload_folder, folder_name)
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

def generate_unique_filename(upload_folder, filename):
    base_name, extension = os.path.splitext(filename)
    unique_name = f"{base_name}_{uuid.uuid4().hex[:6]}{extension}"
    return os.path.join(upload_folder, unique_name).replace("\\", "/")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form['model']

        if file:
            filename = file.filename
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)

            unique_folder = generate_unique_folder(upload_folder)
            file_path = os.path.join(unique_folder, filename).replace("\\", "/")
            file.save(file_path)

            detections = detect_vertebrae(file_path, model_paths[selected_model])

            bounding_boxes_img_path = os.path.join(unique_folder, 'bounding_boxes.jpg').replace("\\", "/")
            save_image_with_bounding_boxes(cv2.imread(file_path), detections, bounding_boxes_img_path)

            middle_points_img_path = os.path.join(unique_folder, 'middle_points.jpg').replace("\\", "/")
            vertebrae_points = save_image_with_middle_points(cv2.imread(file_path), detections, middle_points_img_path)

            cobb_angles_list, cobb_angle_img_path, curve_type = calculate_cobb_angle(vertebrae_points, cv2.imread(middle_points_img_path), middle_points_img_path)
            cobb_angles_list = np.round(cobb_angles_list, 2)

            result = {
                'filename': filename,
                'cobb_angles': cobb_angles_list.tolist(),
                'curve_type': curve_type,
                'image_path': file_path,
                'bounding_boxes_img_path': bounding_boxes_img_path,
                'middle_points_img_path': middle_points_img_path,
                'cobb_angle_img_path': cobb_angle_img_path
            }

            save_result(result)
            update_paths_in_results()

            return render_template('result.html', 
                                   angle_type=curve_type,
                                   cobb_angle=cobb_angles_list, 
                                   image_path=file_path,
                                   bounding_boxes_img_path=bounding_boxes_img_path,
                                   middle_points_img_path=middle_points_img_path,
                                   cobb_angle_img_path=cobb_angle_img_path)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/results')
def results():
    results = load_results()
    results.sort(key=lambda x: x['date'], reverse=True)
    return render_template('previous_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)