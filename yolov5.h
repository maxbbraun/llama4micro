#include <string>
#include <vector>

#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"

namespace yolo {

// The intersection over union threshold used in non-maximum suppression.
const float kNmsIouThreshold = 0.1f;  // Aggressive

// An object detection result.
struct Object {
  std::string label;
  float confidence;
  float x;
  float y;
  float width;
  float height;
};

// Calculates the intersection over union of two objects' bounding boxes.
inline float IntersectionOverUnion(Object& a, Object& b) {
  float intersection_width = std::max(
      0.0f, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
  float intersection_height = std::max(
      0.0f, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
  float intersection_area = intersection_width * intersection_height;
  float union_area =
      a.width * a.height + b.width * b.height - intersection_area;
  return intersection_area / union_area;
}

// Performs non-maximum suppression on a list of objects.
std::vector<Object> NonMaximumSuppression(std::vector<Object>& objects) {
  std::vector<Object> final_objects;

  for (size_t index_a = 0; index_a < objects.size(); ++index_a) {
    Object object_a = objects[index_a];

    // Compare each object to all others to determine whether to keep it.
    bool discard_a = false;
    for (size_t index_b = 0; index_b < objects.size(); ++index_b) {
      Object object_b = objects[index_b];

      // Don't compare the object to itself.
      if (index_a == index_b) {
        continue;
      }

      // Only compare objects with the same label.
      if (object_a.label != object_b.label) {
        continue;
      }

      // Scrutinize object pairs with overlapping bounding boxes.
      if (IntersectionOverUnion(object_a, object_b) > kNmsIouThreshold) {
        // Keep the object if it has the highest confidence.
        if (object_a.confidence > object_b.confidence) {
          continue;
        }

        // Keep the object if confidences are tied and it has a larger area.
        if (object_a.confidence == object_b.confidence &&
            object_a.width * object_a.height >
                object_b.width * object_b.height) {
          continue;
        }

        // Otherwise, discard the object.
        discard_a = true;
        break;
      }
    }

    // Only keep non-discarded objects.
    if (!discard_a) {
      final_objects.push_back(object_a);
    }
  }

  return final_objects;
}

// Dequantizes a quantized value based on the quantization parameters.
inline float Dequantize(uint8_t quantized_value,
                        TfLiteQuantizationParams& quantization_params) {
  return (static_cast<int>(quantized_value) - quantization_params.zero_point) *
         quantization_params.scale;
}

// Processes the output tensor and returns a list of detected objects.
std::vector<Object> GetDetectionResults(tflite::MicroInterpreter* interpreter,
                                        float label_confidence_threshold,
                                        float class_score_threshold,
                                        float min_bbox_size,
                                        std::vector<std::string>* labels) {
  // Extract the data from the output tensor.
  auto output_tensor = interpreter->output_tensor(0);
  const int num_rows = output_tensor->dims->data[1];
  int row_dims = output_tensor->dims->data[2];
  int header_size = 5;  // x, y, width, height, confidence
  uint8_t* data = output_tensor->data.uint8;
  TfLiteQuantizationParams quantization_params = output_tensor->params;

  // Rows come in groups of header_size + num_labels.
  std::vector<Object> raw_results;
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

    // Clip the bounding box to the image.
    x = std::max(0.0f, std::min(x, 1.0f));
    y = std::max(0.0f, std::min(y, 1.0f));
    width = std::max(0.0f, std::min(width, 1.0f - x));
    height = std::max(0.0f, std::min(height, 1.0f - y));

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
    raw_results.push_back(object);
  }

  // Perform naive non-maximum suppression.
  auto filtered_results = NonMaximumSuppression(raw_results);

  // Sort the results by closeness to the center of the image.
  std::sort(filtered_results.begin(), filtered_results.end(),
            [](auto& a, auto& b) {
              float a_horizontal_distance = a.x + a.width / 2 - 0.5f;
              float a_vertical_distance = a.y + a.height / 2 - 0.5f;
              float a_distance_squared =
                  a_horizontal_distance * a_horizontal_distance +
                  a_vertical_distance * a_vertical_distance;
              float b_horizontal_distance = b.x + b.width / 2 - 0.5f;
              float b_vertical_distance = b.y + b.height / 2 - 0.5f;
              float b_distance_squared =
                  b_horizontal_distance * b_horizontal_distance +
                  b_vertical_distance * b_vertical_distance;
              return a_distance_squared < b_distance_squared;
            });

  return filtered_results;
}

}  // namespace yolo
