# MicroExpNet.js
Translation of [MicroExpNet](https://github.com/cuguilke/microexpnet) to JavaScript using [TensorFlow.js](https://js.tensorflow.org/).


## Requirements
- Python 3.6.2
- tensorflow 1.8.0
- tensorflowjs 0.4.2
- Flask 1.0.2 (For serve model file)
- face.png (84x84x3)


## Usage
### 1. Clone repositories
```bash
git clone https://github.com/walkingmask/microexpnetjs.git
cd microexpnetjs
git clone https://github.com/cuguilke/microexpnet.git
cp generator.py microexpnet/
```

### 2. Generate pb file
```bash
cd microexpnet
python generator.py
# this will makes ./output_graph.pb
```

### 3. tensorflowjs-converter
```bash
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Add_1' ./output_graph.pb ../demoapp/static/js/model
```

### 4. Prepar face.png
```bash
cd ..  # microexpnetjs
mkdir demoapp/static/img
cp /path/to/face.png demoapp/static/img/
# size of face.png must be 84x84x3
```

### 5. Run demoapp
```bash
python demoapp/app.py
# You will open http://localhost:5000
```


## Todo
- Add face detection
- As realtime
- GitHub Pages
