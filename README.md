# Indian Cricket Team Face Identification

Pipeline: **MTCNN** (face detect/crop) → **FaceNet** (512-d embedding, pretrained on VGGFace2) → **SVM** classifier.
Pretrained embeddings mean you only need ~30-100 images per player and no GPU.

## 1. Setup
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## 2. Get the dataset
Search Kaggle for "Indian cricketers face dataset" / "cricket players images", or collect images yourself
(Google Dataset Search is also an option). Arrange as (folder name = label):

    dataset/
      Virat_Kohli/   *.jpg
      Rohit_Sharma/  *.jpg
      MS_Dhoni/      *.jpg

With the Kaggle CLI:  `kaggle datasets download -d <owner>/<dataset> --unzip -p raw_data`
then rename/move folders into the layout above. Tips: 5-10 players to start, 30+ images each,
varied angles/lighting, one person per photo, remove wrong/duplicate images.

## 3. Train
    python train.py --data_dir dataset
Prints per-player precision/recall, saves `model.joblib` and `confusion_matrix.png`.
(If you change the dataset, delete `embeddings.npz` first, since it is a cache.)

## 4. Predict
    python predict.py some_photo.jpg
    streamlit run app.py        # web UI with upload

## Notes
- Faces with confidence below `--threshold` are reported as "Unknown", so non-team members aren't forced into a class.
- Images with no detectable face are skipped during training.
- Accuracy depends heavily on dataset cleanliness; check `confusion_matrix.png` to find confusable players.
