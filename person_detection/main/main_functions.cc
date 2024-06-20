/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "driver/gpio.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

#include "tensorflow/lite/micro/kernels/esp_nn/conv_timer.h"
#include "tensorflow/lite/micro/kernels/esp_nn/fully_connected_timer.h"
#include "tensorflow/lite/micro/kernels/esp_nn/pooling_timer.h"
#include "tensorflow/lite/micro/kernels/esp_nn/softmax_timer.h"
#include "tensorflow/lite/micro/kernels/reshape.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 100 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t lata_score = output->data.uint8[kLataIndex];
  int8_t no_lata_score = output->data.uint8[kNotALataIndex];

  float lata_score_f =
      (lata_score - output->params.zero_point) * output->params.scale;
  float no_lata_score_f =
      (no_lata_score - output->params.zero_point) * output->params.scale;

  // Respond to detection
  RespondToDetection(lata_score_f, no_lata_score_f);
  vTaskDelay(1); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  //Para medir el tiempo de inferencia
  long long start_inference_time = esp_timer_get_time();

  // Medir tiempo de cuantización
  long long start_quantization_time = esp_timer_get_time();

  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;

    // printf("%d, ", input->data.int8[i]);
  }

  long long end_quantization_time = esp_timer_get_time();
  long long total_quantization_time = end_quantization_time - start_quantization_time;

  printf("\nTIEMPOS (microsegundos):\n");
  printf("\nTiempo Cuantización: %lld\n", total_quantization_time);
  printf("\nTiempo capas:\n");

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  long long total_conv_time = GetTotalConvTime();
  printf("\nTiempo total Conv2D: %lld\n", total_conv_time);

  long long total_pooling_time = GetTotalPoolingTime();
  printf("Tiempo total MaxPooling: %lld\n", total_pooling_time);
  
  long long total_fullyConnected_time = GetTotalFullyConnectedTime();
  printf("Tiempo total Fully Connected: %lld\n", total_fullyConnected_time);

  long long total_reshape_time = GetTotalReshapeTime();
  printf("Tiempo total Flatten: %lld\n", total_reshape_time);

  long long total_softmax_time = GetTotalSoftmaxTime();
  printf("Tiempo total Softmax: %lld\n", total_softmax_time);

  //Medir tiempo de procesamiento de los resultados
  long long start_result_time = esp_timer_get_time();

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t lata_score = output->data.uint8[kLataIndex];
  int8_t no_lata_score = output->data.uint8[kNotALataIndex];

  float lata_score_f =
      (lata_score - output->params.zero_point) * output->params.scale;
  float no_lata_score_f =
      (no_lata_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(lata_score_f, no_lata_score_f);

  long long end_inference_time = esp_timer_get_time();
  long long total_inference_time = end_inference_time - start_inference_time;

  long long end_result_time = esp_timer_get_time();
  long long result_time = end_result_time - start_result_time;
  printf("Tiempo de procesamiento de respuesta: %lld\n", result_time);

  // printf("Tiempo de inferencia: %lld microsegundos\n", total_inference_time);

  long long sum_subtasks = result_time + total_quantization_time + total_fullyConnected_time + total_pooling_time + total_conv_time + total_reshape_time + total_softmax_time; 
  printf("Suma de los subtasks: %lld\n", sum_subtasks);

  //Tiempo total de inferencia
  printf("\nTiempo Total Inferencia: %lld\n", total_inference_time);

  //Identificar el bottleneck
  const char* bottleneck_operation = "Conv2D";
  long long bottleneck_time = total_conv_time;

  if(total_pooling_time > bottleneck_time){
    bottleneck_operation = "MaxPooling";
    bottleneck_time = total_pooling_time;
  }

  if(total_fullyConnected_time > bottleneck_time){
    bottleneck_operation = "FullyConnected";
    bottleneck_time = total_fullyConnected_time;
  }

  if(total_reshape_time > bottleneck_time){
    bottleneck_operation = "Flatten";
    bottleneck_time = total_reshape_time;
  }

  if(total_softmax_time > bottleneck_time){
    bottleneck_operation = "Softmax";
    bottleneck_time = total_softmax_time;
  }

  if(result_time > bottleneck_time){
    bottleneck_operation = "Procesamiento respuesta";
    bottleneck_time = result_time;
  }

  if(total_quantization_time > bottleneck_time){
    bottleneck_operation = "Cuantización";
    bottleneck_time = total_quantization_time;
  }

  printf("Operacion Bottleneck: %s\n", bottleneck_operation);
  printf("Tiempo Bottleneck: %lld\n", bottleneck_time);

}

