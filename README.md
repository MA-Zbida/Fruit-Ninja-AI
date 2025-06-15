# 🍉 Fruit Ninja V8 – Object Detection & Smart Auto-Player

This project is an under-development AI system designed to **detect fruits, bombs, and game elements in Fruit Ninja**, using a custom-trained **YOLOv8-nano** model and control a **smart slicing system** based on **A* pathfinding*\* for optimized scoring.

##  Project Goals

* Train a lightweight, real-time object detection model for Fruit Ninja.
* Integrate an intelligent auto-slicing strategy using **A* algorithm*\* to cut fruits while avoiding bombs.
* Capture screen frames, detect objects, and control slicing motions to beat the **highest possible score** in the game.

---

##  Project Structure

```bash
fruit-ninja-v8/
├── data/                 # ~1500 labeled images (YOLO format)
│   ├── images/           # Game screenshots
│   └── labels/           # YOLO-format annotations
├── Videos/               # Videos of current Scores
├── src/                  
│   ├── Fruit.py       # Defines the structure of an object
│   ├── Astar.py             # A* pathfinding logic for optimal slicing
│   ├── VideoRecorder.py     # Records the game for future analysis
│   ├── ScreenCapture.py     # Captures game screen in real time
│   ├── main.py              # main script
|   └── requirement.txt      # the libraries used
|   ├── mainGame.py          # All in one code
├── Model YOLOv8n/
│   ├── fruit_ninja_v8.pt       # Trained YOLOv8-nano model
|   ├── Confusion Matrix.jpeg   # Confusion Matrix
|   ├── Metrics.jpeg            # Metrics 
|   ├── Validation.jpeg         # Some Validation Images
├── README.md             # Project overview and setup instructions
```

---

## 🧠 AI Model: YOLOv8-Nano

* **Architecture**: YOLOv8-nano (Ultralytics)
* **Classes**: Fruits (banana, apple, etc.), Bombs, Special items
* **Input size**: 640x640
* **Training Data**: \~1500 images captured from gameplay
* **Annotation Format**: YOLOv5/8-style `.txt` files

### ✅ Evaluation Metrics:

* **Precision, Recall, mAP** almost 1.0 (check Model YOLOv8n/Metrics.png)
* **Confusion matrix** visualized in `/metrics/`


## 🤖 Slicing AI – A\* Path Optimizer

To maximize score while minimizing risk:

* Uses the **A* algorithm*\* to calculate the optimal path through fruit clusters.
* Avoids bombs and plans cuts that maximize combos and bonuses.


## 🖥️ How It Works

1. **Screen Capture & Start Recording if needed**: Continuously grabs game frames.
2. **Detection**: Passes frame to YOLOv8 model.
3. **Analysis**: Finds fruit positions, avoids bombs.
4. **A* Controller*\*: Calculates best swipe path.
5. **Input Trigger**: Sends slicing gesture via system mouse/touch input.


## 🚧 Under Development

* [ ] Improving bomb avoidance logic
* [ ] Using a faster Screen Capture instead of mss

##  Run the Project

Clone the repository
```bash
git clone https://github.com/MA-Zbida/Fruit-Ninja-AI.git
cd Fruit-Nnja-AI
```
**Make sure you have fruit ninja installed**
```bash
pip install -r src/requirements.txt
```
**run this when you want to load up the game**
```bash
python src/main.py 
```
or
```bash
python src/mainGame.py
```


## 📬 Contact

For questions or collaboration: 
**\[ABDERRAZAK KHALIL] – AI Engineering Student** 
**\[ZBIDA MOHAMMED AMINE ] – AI Engineering Student** 

📧 Email: \[[Abderrazak Khalil](mailto:khalilabderrazak1@gmail.com)]
📧 Email: \[[Mohamed Amine Zbida](mailto:itzzbida@gmail.com)]


