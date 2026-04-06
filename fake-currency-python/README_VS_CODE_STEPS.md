# Fake Currency Detection – Python (Streamlit) Prototype

This is a **Python-only** prototype. It uses **Streamlit** for the UI and **OpenCV** heuristics for a quick demo.
Later you can replace the heuristic with your **CNN model** (TensorFlow / PyTorch).

---

## 🧰 Prerequisites (install once)
- **Python 3.10+**
- **Visual Studio Code**

---

## 🧭 Step-by-step in VS Code

1. **Create a folder and open it**
   - Make a folder, e.g., `fake-currency-python`
   - Open VS Code → *File → Open Folder…* → select it

2. **Open a terminal in VS Code**
   - *Terminal → New Terminal*

3. **(Recommended) Create & activate a virtual environment**
   - Windows (PowerShell):
     ```ps1
     python -m venv .venv
     .venv\Scripts\Activate
     ```
   - macOS / Linux (bash/zsh):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. **Copy the project files here**
   - Put `app.py`, `detection.py`, and `requirements.txt` in your project folder.
   - (If you downloaded the ZIP, just extract it here.)

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Your browser opens to the app. If not, go to the URL shown in the terminal (usually http://localhost:8501).

7. **Demo flow**
   - Click **Browse files** and select any banknote image (JPG/PNG).
   - You’ll see the uploaded image, a **Result label**, a **Confidence score**, and **feature details**.
   - This is a **heuristic demo** (not a real detector).

---

## 🧪 How the heuristic works (for your presentation)
- **Sharpness** (Variance of Laplacian): checks if the image is in focus to avoid bad inputs.
- **Edge density** (Canny edges): genuine notes often show structured edges; the score prefers moderate density.
- **Brightness mean**: avoids too dark/bright photos.

We combine these into a single score and map to:
- **Likely Genuine** (score ≥ 0.5)
- **Potentially Fake** (score < 0.5)

> ⚠️ This is **not accurate** for real-world detection. Replace with a trained CNN when available.

---

## 🔁 Replacing heuristic with your CNN (later)
- Export your trained model (TensorFlow SavedModel / PyTorch `.pt`).
- In `analyze_note()`:
  - Preprocess the image to your model’s input size
  - Load and run the model
  - Convert the prediction to label + confidence
- Keep the Streamlit UI the same.

---

## ❓ Common issues
- **`cv2` import error** → Ensure `pip install -r requirements.txt` ran successfully.
- **No browser opened** → Copy the URL from the terminal and paste into your browser.
- **Permission errors on macOS** → Try `python3 -m venv .venv` and `pip3 install -r requirements.txt`.

---

## 🗣️ Presentation script (quick)
- “I built a **Python-based** Fake Currency Detector prototype.”
- “The **Streamlit UI** handles upload, shows results and features.”
- “Right now it's a **heuristic** using OpenCV for demo. Next, we’ll plug in our **CNN** for real detection and add a **FastAPI** backend for mobile/web clients.”
