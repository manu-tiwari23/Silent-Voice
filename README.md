🧤 Silent Voice – Hand Gesture Recognition to Speech System
Silent Voice is an AI-powered Python project that recognizes hand gestures through flex sensor readings and converts them into spoken words using machine learning and text-to-speech. It enables non-verbal or speech-impaired individuals to communicate using sign language, which is translated into audio output in real time.

🚀 Features
🔤 Recognizes hand gestures corresponding to A–Z alphabets

🧠 Trained using a Random Forest Classifier on simulated sensor data

📖 Maps sequences of gestures to valid English words

🔊 Uses Google Text-to-Speech (gTTS) to speak valid words

🧪 CLI interface for both training and testing

✅ Contains a growing list of 100+ common daily-use words

🎯 Motivation
Many individuals struggle to communicate due to speech disabilities. Silent Voice bridges this communication gap by providing a low-cost and real-time system to translate hand gestures into voice using AI and electronics.

🛠️ How It Works
Sensor Input: Five finger flex sensor values are entered (simulating glove input).

Gesture Prediction: ML model predicts the corresponding alphabet.

Word Formation: User inputs multiple gestures to form a word.

Validation: Word is checked against a list of commonly used words.

Speech Output: If valid, the word is converted to speech using gTTS and played.

📦 Requirements
bash
Copy
Edit
pip install numpy scikit-learn gTTS playsound joblib
▶️ Usage
Run the code:

bash
Copy
Edit
python silent_voice.py
Choose options:

Train model

Test gestures for words (input 5 flex values per alphabet)

If a valid word is recognized, it will also be spoken aloud by your system.

💡 Future Scope
Integrate with real-time flex sensor gloves (using ESP32, etc.)

Add webcam-based hand pose recognition

Expand dictionary and support sentence formation

🤝 Contribution
PRs and feedback are welcome! Let’s work together to empower silent communication through tech.
