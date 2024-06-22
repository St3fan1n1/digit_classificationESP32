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

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/mcpwm.h"
#include "soc/mcpwm_periph.h"

#define SERVO_PIN 18


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

// Function to initialize the MCPWM module for controlling the servo
void mcpwm_example_gpio_initialize(void) {
    printf("Initializing MCPWM servo control...\n");
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, SERVO_PIN);
}

void servo_control(mcpwm_unit_t mcpwm_num, mcpwm_timer_t timer_num, float angle) {
    // Calculate pulse width (500us - 2500us) corresponding to the angle (-90° - 270°)
    uint32_t duty_us = (500 + ((angle + 90) / 360.0) * 2000);
    mcpwm_set_duty_in_us(mcpwm_num, timer_num, MCPWM_OPR_A, duty_us);
}

void RespondToDetection(float lata_score, float no_lata_score) {
  // Initialize MCPWM
  mcpwm_example_gpio_initialize();

  // Configure MCPWM
  mcpwm_config_t pwm_config;
  pwm_config.frequency = 50;    // Frequency = 50Hz, for servo motor
  pwm_config.cmpr_a = 0;        // Duty cycle of PWMxA = 0
  pwm_config.cmpr_b = 0;        // Duty cycle of PWMxB = 0
  pwm_config.counter_mode = MCPWM_UP_COUNTER;
  pwm_config.duty_mode = MCPWM_DUTY_MODE_0;
  mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &pwm_config);

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
    // gpio_set_level(BLINK_GPIO, 1);
    for (int angle=-90; angle<=270;angle++){
      servo_control(MCPWM_UNIT_0, MCPWM_TIMER_0, angle);
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  } else {
    for (int angle=270; angle>=-90; angle--){
      servo_control(MCPWM_UNIT_0, MCPWM_TIMER_0, angle);
      vTaskDelay(pdMS_TO_TICKS(10));
    }
    // gpio_set_level(BLINK_GPIO, 0);
  }
}
