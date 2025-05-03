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
├── src/                  
│   ├── detector.py       # YOLOv8 nano inference and processing
│   ├── astar.py          # A* pathfinding logic for optimal slicing
│   ├── controller.py     # Controls the slicing motions (mouse/touch)
│   ├── screen_capture.py # Captures game screen in real time
│   └── utils.py          # Helper functions
├── model/
│   └── fruitninja_v8.pt  # Trained YOLOv8-nano model
├── metrics/
│   └── confusion_matrix.png  # Visual metrics and evaluation results
├── README.md             # Project overview and setup instructions
```

---

## 🧠 AI Model: YOLOv8-Nano

* **Architecture**: YOLOv8-nano (Ultralytics)
* **Classes**: Fruits (banana, apple, etc.), Bombs, Special items
* **Input size**: 320x320
* **Training Data**: \~1500 images captured from gameplay
* **Annotation Format**: YOLOv5/8-style `.txt` files

### ✅ Evaluation Metrics:

* **Precision, Recall, mAP** at 0.5 and 0.5:0.95
* **Confusion matrix** visualized in `/metrics/`


## 🤖 Slicing AI – A\* Path Optimizer

To maximize score while minimizing risk:

* Uses the **A* algorithm*\* to calculate the optimal path through fruit clusters.
* Avoids bombs and plans cuts that maximize combos and bonuses.


## 🖥️ How It Works

1. **Screen Capture**: Continuously grabs game frames.
2. **Detection**: Passes frame to YOLOv8 model.
3. **Analysis**: Finds fruit positions, avoids bombs.
4. **A* Controller*\*: Calculates best swipe path.
5. **Input Trigger**: Sends slicing gesture via system mouse/touch input.


## 🚧 Under Development

* [ ] Improved bomb avoidance logic
* [ ] Enhanced A\* slicing in real-time
* [ ] Game speed adaptation
* [ ] Full automation to beat high scores

## 📌 Requirements

* Python 3.8+
* `ultralytics`, `opencv-python`, `numpy`, `pyautogui`, `mss`, etc.

Install dependencies:


##  Run the Project



## 📬 Contact

For questions or collaboration: **\[ABDERRAZAK KHALIL] – AI Engineer** & **\[ZBIDA MOHAMMED AMINE ] – AI Engineer** 
📧 Email: \[[your\_email@example.com](mailto:khalilabderrazak1@gmail.com)]

