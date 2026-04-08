set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(GCC_PATH /opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin)
set(GCC_COMPILER ${GCC_PATH}/aarch64-linux-gnu)
set(CMAKE_C_COMPILER ${GCC_COMPILER}-gcc)
set(CMAKE_CXX_COMPILER ${GCC_COMPILER}-g++)