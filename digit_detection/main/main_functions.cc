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
constexpr int scratchBufSize = 40 * 1024;
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
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMaxPool2D();


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
  // if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
  //   MicroPrintf("Image capture failed.");
  // }

  // // Run the model on this input and make sure it succeeds.
  // if (kTfLiteOk != interpreter->Invoke()) {
  //   MicroPrintf("Invoke failed.");
  // }

  // TfLiteTensor* output = interpreter->output(0);

  // // Process the inference results.
  // int8_t person_score = output->data.uint8[kPersonIndex];
  // int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  // float person_score_f =
  //     (person_score - output->params.zero_point) * output->params.scale;
  // float no_person_score_f =
  //     (no_person_score - output->params.zero_point) * output->params.scale;

  // // Respond to detection
  // RespondToDetection(person_score_f, no_person_score_f);
  // vTaskDelay(1); // to avoid watchdog trigger

  if (kTfLiteOk != interpreter->Invoke()){
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // const int num_classes = 10;
  // float scores[num_classes];
  // for(int i = 0; i<num_classes; i++){
  //   scores[i] = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;
  // }


  // int predicted_digit = 0;
  // float max_score = scores[0];
  // for (int i = 1; i < num_classes; ++i) {
  //   if (scores[i] > max_score) {
  //     max_score = scores[i];
  //     predicted_digit = i;
  //   }
  // }

  RespondToDetection(output->data.f, output->dims->data[1]);

  vTaskDelay(1);

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
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    // input->data.int8[i] = ((uint8_t *) ptr)[i];
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;

    printf("%d, ", input->data.int8[i]);
  }
  printf("\n");

  for(int i = 0; i<10; i++){
    printf("Input[%d]: %d\n", i, input->data.int8[i]);
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);
  // float* output_data = output->data.f;

  if (output->type == kTfLiteInt8){
    int8_t* output_data = output->data.int8;
    int length = output->dims->data[1];

    printf("Output tensor length: %d\n", length);


    
    // RespondToDetection(output_data, length)
    
    
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    int max_index = 0;
    float max_value = (output_data[0] - zero_point) * scale;

     for(int i = 0; i<length; i++){
      float value = (output_data[i] - zero_point) * scale;

      printf("Output[%d]: %f\n", i, value);
      if (value > max_value){
        max_value = value;
        max_index = i;
      }
     }

     MicroPrintf("Detected digit: %d with confidence %.2f%%", max_index, max_value*100);
    //  RespondToDetection(output_data, length);
  } else {
    MicroPrintf("Unexpected output tensor type: %d", output->type);
  }

  // Process the inference results.
  // int8_t person_score = output->data.uint8[kPersonIndex];
  // int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  // float person_score_f =
  //     (person_score - output->params.zero_point) * output->params.scale;
  // float no_person_score_f =
  //     (no_person_score - output->params.zero_point) * output->params.scale;
  // RespondToDetection(output_data, output->dims->data[1]);
}
