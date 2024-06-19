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
#define IMG_WD (96 * 2)
#define IMG_HT (96 * 2)

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *lata_indicator = NULL;
static lv_obj_t *label = NULL;

static void create_gui(void)
{
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  lata_indicator = lv_led_create(lv_scr_act());
  assert(lata_indicator);
  lv_obj_align(lata_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  lv_led_set_color(lata_indicator, lv_palette_main(LV_PALETTE_GREEN));

  label = lv_label_create(lv_scr_act());
  assert(label);
  lv_label_set_text_static(label, "Lata detected");
  lv_obj_align_to(label, lata_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
  bsp_display_unlock();
}
#endif // DISPLAY_SUPPORT

void RespondToDetection(float lata_score, float no_lata_score) {
  int lata_score_int = (lata_score) * 100 + 0.5;
  (void) no_lata_score; // unused
#if DISPLAY_SUPPORT
    if (!camera_canvas) {
      create_gui();
    }

    uint16_t *buf = (uint16_t *) image_provider_get_display_buf();

    bsp_display_lock(0);
    if (lata_score_int < 60) { // treat score less than 60% as no person
      lv_led_off(lata_indicator);
    } else {
      lv_led_on(lata_indicator);
    }
    lv_canvas_set_buffer(camera_canvas, buf, IMG_WD, IMG_HT, LV_IMG_CF_TRUE_COLOR);
    bsp_display_unlock();
#endif // DISPLAY_SUPPORT
  MicroPrintf("lata score:%d%%, no lata score %d%%",
              lata_score_int, 100 - lata_score_int);

  if (lata_score_int > 60){
    gpio_set_level(BLINK_GPIO, 1);
  } else {
    gpio_set_level(BLINK_GPIO, 0);
  }
}
