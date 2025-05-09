import cv2
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATASET_PATH = 'dataset'
MODEL_PATH = 'keras_model.h5'

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ Error: Could not load Haar Cascade. Check the file path.")
    exit()

# === Collect Face Images ===
def collect_faces():
    print("📸 Face Data Collection")
    name = input("Enter Your Name: ").strip()

    person_dir = os.path.join(DATASET_PATH, name)
    while os.path.exists(person_dir):
        print("⚠️ Name already exists. Try a different name.")
        name = input("Enter Your Name Again: ").strip()
        person_dir = os.path.join(DATASET_PATH, name)

    os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    print("➡ Look at the camera. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            print(f"✅ Saved: {file_path}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done. Collected {count} images for '{name}'.")

# === Train the Model ===
def train_model():
    print("🧠 Starting training...")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(100, 100),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(100, 100),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_data.class_indices)
    if num_classes < 2 or train_data.samples == 0:
        print("❌ Not enough data to train. Collect more images for at least 2 people.")
        return

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)

    model.save(MODEL_PATH)
    print("✅ Model trained and saved as 'keras_model.h5'.")

# === Real-Time Face Recognition ===
def recognize_faces():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Please train it first.")
        return

    print("🎥 Starting real-time face recognition...")
    model = load_model(MODEL_PATH)
    class_names = sorted(os.listdir(DATASET_PATH))

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            pred = model.predict(face)
            name = class_names[np.argmax(pred)]
            prob = np.max(pred)

            label = f"{name} ({prob*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Recognition stopped.")

# === Main Menu ===
def main():
    print("\n🎯 Face Recognition System")
    print("1. Collect Face Images")
    print("2. Train Model")
    print("3. Recognize Faces")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        collect_faces()
    elif choice == '2':
        train_model()
    elif choice == '3':
        recognize_faces()
    else:
        print("❌ Invalid Choice")

if __name__ == "__main__":
    main()
