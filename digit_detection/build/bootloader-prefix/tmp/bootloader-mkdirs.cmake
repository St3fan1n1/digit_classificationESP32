# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/stefano/esp/idf/esp-idf/components/bootloader/subproject"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/tmp"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/src/bootloader-stamp"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/src"
  "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/stefano/esp/projects_tf/digit_detection/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
