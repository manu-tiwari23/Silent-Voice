import random
import joblib
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dictionary of estimated sensor values (5 fingers) for each alphabet gesture
GESTURES = {
    'A': [4000, 1000, 1000, 1000, 1000],
    'B': [1000, 4000, 4000, 4000, 4000],
    'C': [4000, 4000, 4000, 4000, 4000],
    'D': [1000, 4000, 1000, 1000, 1000],
    'E': [4000, 2000, 2000, 2000, 2000],
    'F': [4000, 2000, 2000, 1000, 1000],
    'G': [3900, 3900, 1000, 1000, 1000],
    'H': [1000, 4000, 4000, 1000, 1000],
    'I': [1000, 1000, 1000, 1000, 4000],
    'J': [1000, 1000, 1000, 1200, 4000],
    'K': [4000, 4000, 2000, 1000, 1000],
    'L': [4000, 4000, 1000, 1000, 1000],
    'M': [4000, 4000, 4000, 1000, 1000],
    'N': [4000, 4000, 1000, 1000, 4000],
    'O': [3700, 3700, 3700, 3700, 3700],
    'P': [4000, 4000, 4000, 1000, 1000],
    'Q': [4000, 4000, 2000, 2000, 2000],
    'R': [2000, 4400, 4000, 1000, 1000],
    'S': [4200, 4200, 4200, 4200, 4200],
    'T': [4000, 1000, 4000, 4000, 4000],
    'U': [1000, 4000, 4000, 1000, 1000],
    'V': [1000, 4000, 4000, 1000, 1100],
    'W': [1000, 4000, 4000, 4000, 1000],
    'X': [4000, 2000, 1000, 1000, 1000],
    'Y': [4000, 1000, 1000, 1000, 4000],
    'Z': [4000, 4000, 1000, 4000, 1000]
}

VALID_WORDS = {
    "HI", "BYE", "YES", "NO", "OK", "HELP", "LOVE", "YOU",
    "GO", "RUN", "STOP", "START", "EAT", "DRINK", "WATER", "FOOD", "HOME", "FIRE",
    "SAVE", "SIT", "STAND", "COME", "WAIT", "CALL", "DANCE", "OPEN", "CLOSE", "LOOK",
    "SEE", "TALK", "LISTEN", "WALK", "TURN", "RIGHT", "LEFT", "UP", "DOWN", "IN", "OUT",
    "GIVE", "TAKE", "BRING", "CARRY", "THROW", "CATCH", "HOLD", "READ", "WRITE", "SLEEP",
    "WAKE", "EYE", "HAND", "LEG", "HEAD", "DOOR", "WINDOW", "CHAIR", "TABLE", "BED", "ROOM",
    "BOOK", "PEN", "BAG", "SHIRT", "PANTS", "SHOES", "LIGHT", "FAN", "TV", "PHONE", "CLOCK",
    "SCHOOL", "COLLEGE", "WORK", "PLAY", "STUDY", "TEACH", "LEARN", "LAUGH", "CRY", "SMILE",
    "ANGRY", "HAPPY", "TIRED", "SAD", "HUNGRY", "THIRSTY", "FAST", "SLOW", "BIG", "SMALL",
    "HOT", "COLD", "GOOD", "BAD", "NEW", "OLD", "NEAR", "FAR", "CLEAN", "DIRTY"
}

MODEL_FILENAME = "gesture_model.pkl"

def add_noise(values, noise_range=200):
    return [v + random.randint(-noise_range, noise_range) for v in values]

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")

def train_model():
    print("\n🎯 Training model...")

    X, y = [], []
    for letter, readings in GESTURES.items():
        for _ in range(5):
            X.append(add_noise(readings))
            y.append(letter)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, clf.predict(X_test)))

    joblib.dump(clf, MODEL_FILENAME)
    print(f"✅ Model trained and saved as '{MODEL_FILENAME}'\n")

def test_gestures():
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Trained model file '{MODEL_FILENAME}' not found. Please train it first.\n")
        return

    try:
        n = int(input("\n🔢 How many gestures (alphabets) do you want to input? "))
    except:
        print("❌ Invalid number.")
        return

    letters = []

    for i in range(n):
        raw = input(f"✋ Enter 5 finger values for gesture {i+1} (space-separated): ")
        try:
            vals = list(map(int, raw.strip().split()))
            if len(vals) != 5:
                print("❌ Invalid input! Please enter exactly 5 integers.")
                continue
            vals_array = np.array(vals).reshape(1, -1)
            predicted_letter = model.predict(vals_array)[0]
            print(f"✅ Predicted letter: {predicted_letter}")
            letters.append(predicted_letter)
        except Exception as e:
            print("❌ Error reading values:", e)

    final = ''.join(letters).upper()
    print("\n🔎 Final Sequence:", final)

    if final in VALID_WORDS:
        print("✅ Recognized Word:", final)
        speak(final)  # 🗣️ Speak the word
    else:
        print("🔤 Recognized Letters (not a valid word):", ' '.join(letters))

def main():
    while True:
        print("\n🧠 AI Hand Gesture System")
        print("1. Train Model")
        print("2. Test Gestures for Words")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            train_model()
        elif choice == '2':
            test_gestures()
        elif choice == '3':
            print("👋 Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
