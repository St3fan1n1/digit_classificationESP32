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

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "driver/gpio.h"
#define BLINK_GPIO GPIO_NUM_4

#include "esp_main.h"
#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp-bsp.h"

// Camera definition is always initialized to match the trained detection model: 96x96 pix
// That is too small for LCD displays, so we extrapolate the image to 192x192 pix
#define IMG_WD (28 * 2)
#define IMG_HT (28 * 2)

static lv_obj_t *camera_canvas = NULL;
// static lv_obj_t *person_indicator = NULL;
static lv_obj_t *digit_label = NULL;

static void create_gui(void)
{
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  // person_indicator = lv_led_create(lv_scr_act());
  // assert(person_indicator);
  // lv_obj_align(person_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  // lv_led_set_color(person_indicator, lv_palette_main(LV_PALETTE_GREEN));

  digit_label = lv_label_create(lv_scr_act());
  assert(digit_label);
  // lv_label_set_text_static(label, "Person detected");
  lv_obj_align_to(digit_label, LV_ALIGN_BOTTOM_MID, 0, 0);
  bsp_display_unlock();
}
#endif // DISPLAY_SUPPORT

void RespondToDetection(int8_t* output, int length) {
  // int person_score_int = (person_score) * 100 + 0.5;
  // (void) no_person_score; // unused

  // int max_index = 0;
  // float max_value = output[0];

  // printf("Output[0]: %f\n", max_value);

  // for(int i = 1; i < length; i++){
  //   printf("Output[%d]: %d\n", i, output[i]);
  //   if (output[i] > max_value){
  //     max_value = output[i];
  //     max_index = i;
  //   }
  // }
    // float scale = output->params.scale;
    // int zero_point = output->params.zero_point;

    // int max_index = 0;
    // float max_value = (output_data[0] - zero_point) * scale;

    //  for(int i = 1; i<length; i++){
    //   float value = (output_data[i] - zero_point) * scale;

    //   printf("Output[%d]: %f\n", i, value);
    //   if (value > max_value){
    //     max_value = value;
    //     max_index = i;
    //   }
    //  }

  


#if DISPLAY_SUPPORT
    if (!camera_canvas) {
      create_gui();
    }

    uint16_t *buf = (uint16_t *) image_provider_get_display_buf();

    bsp_display_lock(0);
    // if (person_score_int < 60) { // treat score less than 60% as no person
    //   lv_led_off(person_indicator);
    // } else {
    //   lv_led_on(person_indicator);
    // }
    lv_canvas_set_buffer(camera_canvas, buf, IMG_WD, IMG_HT, LV_IMG_CF_TRUE_COLOR);
    char digit_text[20];
    snprintf(digit_text, sizeof(digit_text), "Digit: %d", max_index);
    lv_label_set_text(digit_label, digit_text);
    bsp_display_unlock();
#endif // DISPLAY_SUPPORT

  // float max_confidence_value = 1.0;
  // float confidence_percentage = max_value  * 100;

  // MicroPrintf("Detected digit: %d with confidence %.2f%%", max_index, max_value * 100);
}
