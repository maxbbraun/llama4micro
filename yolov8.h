#pragma once

#include <algorithm>
#include <cmath>
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

        // Break confidence ties by area, then prefer the earlier input.
        if (object_a.confidence == object_b.confidence) {
          const float area_a = object_a.width * object_a.height;
          const float area_b = object_b.width * object_b.height;
          if (area_a > area_b || (area_a == area_b && index_a < index_b)) {
            continue;
          }
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

// Decodes one box edge from its 16-bin distribution, in grid-cell units.
float DecodeDistance(const uint8_t* logits,
                     const TfLiteQuantizationParams& quantization_params) {
  const uint8_t maximum = *std::max_element(logits, logits + 16);
  float total = 0.0f;
  float weighted_total = 0.0f;
  for (int bin = 0; bin < 16; ++bin) {
    float probability = std::exp((static_cast<int>(logits[bin]) - maximum) *
                                 quantization_params.scale);
    total += probability;
    weighted_total += bin * probability;
  }
  return weighted_total / total;
}

// Decodes the six raw YOLOv8 detection heads and returns detected objects.
std::vector<Object> GetDetectionResults(tflite::MicroInterpreter* interpreter,
                                        float confidence_threshold,
                                        float min_bbox_size,
                                        std::vector<std::string>* labels) {
  // Output indices pair box distributions with class scores at each scale.
  const int box_outputs[] = {4, 5, 0};
  const int class_outputs[] = {1, 3, 2};
  std::vector<Object> raw_results;
  for (int scale = 0; scale < 3; ++scale) {
    auto* boxes = interpreter->output_tensor(box_outputs[scale]);
    auto* classes = interpreter->output_tensor(class_outputs[scale]);
    const int grid_size = classes->dims->data[1];
    const int num_labels = classes->dims->data[3];
    for (int row = 0; row < grid_size * grid_size; ++row) {
      // Class probabilities already include sigmoid; there is no objectness.
      const uint8_t* scores = classes->data.uint8 + row * num_labels;
      const int label = std::max_element(scores, scores + num_labels) - scores;
      float confidence = Dequantize(scores[label], classes->params);
      if (confidence < confidence_threshold) {
        continue;
      }

      // Each grid point predicts left, top, right, and bottom distances.
      float distance[4];
      for (int edge = 0; edge < 4; ++edge) {
        distance[edge] = DecodeDistance(
            boxes->data.uint8 + row * 64 + edge * 16, boxes->params);
      }
      float center_x = row % grid_size + 0.5f;
      float center_y = row / grid_size + 0.5f;
      float x = std::clamp((center_x - distance[0]) / grid_size, 0.0f, 1.0f);
      float y = std::clamp((center_y - distance[1]) / grid_size, 0.0f, 1.0f);
      float right =
          std::clamp((center_x + distance[2]) / grid_size, 0.0f, 1.0f);
      float bottom =
          std::clamp((center_y + distance[3]) / grid_size, 0.0f, 1.0f);
      float width = right - x;
      float height = bottom - y;

      // Discard small bounding boxes. Both sides have to be large enough.
      if (width < min_bbox_size || height < min_bbox_size) {
        continue;
      }
      raw_results.push_back(
          {labels->at(label), confidence, x, y, width, height});
    }
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
