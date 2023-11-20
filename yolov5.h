#include <string>
#include <vector>

#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"

namespace yolo {

// An object detection result.
struct Object {
  std::string label;
  float confidence;
  float x;
  float y;
  float width;
  float height;
};

// Dequantizes a quantized value based on the quantization parameters.
inline float Dequantize(uint8_t quantized_value,
                        TfLiteQuantizationParams quantization_params) {
  return (static_cast<int>(quantized_value) - quantization_params.zero_point) *
         quantization_params.scale;
}

// Processes the output tensor and returns a list of detected objects.
std::vector<Object> GetDetectionResults(tflite::MicroInterpreter* interpreter,
                                        float label_confidence_threshold,
                                        float class_score_threshold,
                                        float min_bbox_size,
                                        std::vector<std::string>* labels) {
  std::vector<Object> results;

  // Extract the data from the output tensor.
  auto output_tensor = interpreter->output_tensor(0);
  const int num_rows = output_tensor->dims->data[1];
  int row_dims = output_tensor->dims->data[2];
  int header_size = 5;  // x, y, width, height, confidence
  uint8_t* data = output_tensor->data.uint8;
  TfLiteQuantizationParams quantization_params = output_tensor->params;

  // Rows come in groups of header_size + num_labels.
  for (int row = 0; row < num_rows; ++row) {
    // The first number is the confidence.
    float confidence = Dequantize(data[row * row_dims], quantization_params);

    // Discard low confidence rows.
    if (confidence < label_confidence_threshold) {
      continue;
    }

    // The next four numbers are the bounding box.
    float x = Dequantize(data[row * row_dims + 1], quantization_params);
    float y = Dequantize(data[row * row_dims + 2], quantization_params);
    float width = Dequantize(data[row * row_dims + 3], quantization_params);
    float height = Dequantize(data[row * row_dims + 4], quantization_params);

    // The remaining numbers are the label scores. Pick the highest one.
    float max_score = 0.0f;
    int max_score_label = 0;
    int num_labels = row_dims - header_size;
    for (int label = 0; label < num_labels; ++label) {
      float score = Dequantize(data[row * row_dims + header_size + label],
                               quantization_params);
      if (score > max_score) {
        max_score = score;
        max_score_label = label;
      }
    }

    // Discard low score classes.
    if (max_score < class_score_threshold) {
      continue;
    }

    // Discard small bounding boxes. Both sides have to be large enough.
    if (width < min_bbox_size || height < min_bbox_size) {
      continue;
    }

    // Assemble the result.
    Object object;
    object.label = labels->at(max_score_label);
    object.confidence = confidence;
    object.x = x;
    object.y = y;
    object.width = width;
    object.height = height;
    results.push_back(object);
  }

  // TODO: Implement non-maximum suppression.

  // Sort the results by closeness to the center.
  std::sort(results.begin(), results.end(), [](auto& a, auto& b) {
    float a_center_dist_sq =
        (a.x - 0.5f) * (a.x - 0.5f) + (a.y - 0.5f) * (a.y - 0.5f);
    float b_center_dist_sq =
        (b.x - 0.5f) * (b.x - 0.5f) + (b.y - 0.5f) * (b.y - 0.5f);
    return a_center_dist_sq < b_center_dist_sq;
  });

  return results;
}

}  // namespace yolo
