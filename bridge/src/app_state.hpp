#pragma once

// 进程级开关：Ctrl-C / SIGINT 置位，各个循环（主循环、replay、capture）都看它。
// 定义留在 main.cpp（唯一的 owner），其他模块只 extern 引用。

#include <atomic>

extern std::atomic<bool> g_stop;
