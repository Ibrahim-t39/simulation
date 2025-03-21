import os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import datetime
import json
import sqlite3
import geopandas as gpd
import folium

# 1. Generate synthetic pipe images
def generate_pipe_images(num_images=10):
    if not os.path.exists("simulated_pipes"):
        os.makedirs("simulated_pipes")
    for i in range(num_images):
        img = Image.new("RGB", (100, 100), color="black")
        draw = ImageDraw.Draw(img)
        label = "lead" if i % 2 == 0 else "non_lead"
        draw.rectangle([20, 20, 80, 80], fill="gray" if label == "lead" else "orange")
        img.save(f"simulated_pipes/pipe_{i}_{label}.png")

# 2. Train CV model on synthetic data
def train_cv_model():
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        "simulated_pipes", target_size=(100, 100), batch_size=2, class_mode="binary"
    )
    base_model = tf.keras.applications.MobileNetV2(input_shape=(100, 100, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, epochs=5)
    model.save("pipe_classifier.h5")

# 3. Simulate field worker image capture
def simulate_image_capture(image_path):
    metadata = {
        "image_path": image_path,
        "gps": {"lat": 35.7796, "lon": -78.6382},
        "timestamp": str(datetime.datetime.now()),
        "inspector": "John Doe"
    }
    with open("upload.json", "w") as f:
        json.dump(metadata, f)
    return metadata

# 4. Analyze image with CV model
def analyze_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100)) / 255.0
    img = np.expand_dims(img, axis=0)
    model = tf.keras.models.load_model("pipe_classifier.h5")
    prediction = model.predict(img)[0][0]
    return "lead" if prediction > 0.5 else "non_lead"

# 5. Validate against historical data
def setup_historical_db():
    conn = sqlite3.connect("historical_pipes.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS pipes (id INTEGER PRIMARY KEY, location TEXT, material TEXT)")
    cursor.executemany("INSERT INTO pipes (location, material) VALUES (?, ?)", [
        ("35.7796,-78.6382", "lead"),
        ("35.7800,-78.6390", "non_lead")
    ])
    conn.commit()
    return conn

def validate_prediction(gps, prediction, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT material FROM pipes WHERE location = ?", (f"{gps['lat']},{gps['lon']}",))
    historical = cursor.fetchone()
    if historical and historical[0] != prediction:
        return "flagged_for_review"
    return "verified"

# 6. Simulate integration and reporting
def send_to_120water(data):
    print("Sent to 120Water:", data)

def update_arcgis(gps, material):
    m = folium.Map(location=[gps["lat"], gps["lon"]], zoom_start=15)
    folium.Marker([gps["lat"], gps["lon"]], popup=material).add_to(m)
    m.save("arcgis_map.html")
    print("ArcGIS updated")

# 7. Feedback loop (placeholder)
def log_feedback(image_path, prediction, correct_label):
    print(f"Feedback logged: {image_path}, Predicted: {prediction}, Correct: {correct_label}")

# Run the simulation
if __name__ == "__main__":
    # Step 1: Generate data
    generate_pipe_images(10)
    print("Generated synthetic images")

    # Step 2: Train model
    train_cv_model()
    print("Trained CV model")

    # Step 3: Simulate capture
    metadata = simulate_image_capture("simulated_pipes/pipe_0_lead.png")
    print("Captured:", metadata)

    # Step 4: Analyze image
    result = analyze_image(metadata["image_path"])
    print("Prediction:", result)

    # Step 5: Validate
    conn = setup_historical_db()
    status = validate_prediction(metadata["gps"], result, conn)
    print("Status:", status)
    conn.close()

    # Step 6: Integrate and report
    send_to_120water({"location": metadata["gps"], "material": result, "status": status})
    update_arcgis(metadata["gps"], result)

    # Step 7: Feedback
    log_feedback(metadata["image_path"], result, "lead")
    print("Simulation complete")