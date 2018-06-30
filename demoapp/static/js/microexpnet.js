const CLASSES = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
const MODEL_FILE_URL = 'static/js/model/tensorflowjs_model.pb'
const WEIGHT_MANIFEST_FILE_URL = 'static/js/model/weights_manifest.json'
const INPUT_NODE_NAME = 'Placeholder';
const OUTPUT_NODE_NAME = 'Add_1';
class MicroExpNet {
  constructor() {}
  async load() {
    this.model = await tf.loadFrozenModel(
      MODEL_FILE_URL,
      WEIGHT_MANIFEST_FILE_URL
    );
  }
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
  predict(input) {
    const reshapedInput = input.reshape([1, 84*84]);
    return this.model.execute(
      {[INPUT_NODE_NAME]: reshapedInput}, OUTPUT_NODE_NAME);
  }
  getTopKClass(logits) {
    const predictions = tf.tidy(() => {
      return logits.argMax(1);
    });
    const values = predictions.dataSync();
    predictions.dispose();
    return CLASSES[values[0]];
  }
}