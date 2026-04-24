/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "utils/logger.h"
#include <iostream>
#include <ctime>
#include <filesystem>

namespace GryFlux
{
    Logger::Logger()
        : currentLevel_(LogLevel::INFO),
          outputTarget_(LogOutputType::BOTH),
          showTimestamp_(true),
          showLogLevel_(true),
          app_name_("Grifcc")
    {
        // 尝试创建默认日志文件
        try
        {
            // 确保/var/log目录存在
            std::filesystem::path dirPath("/var/log");
            if (!std::filesystem::exists(dirPath))
            {
                try
                {
                    std::filesystem::create_directories(dirPath);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "无法创建日志目录: " << e.what() << std::endl;
                }
            };

            // 打开日志文件
            if (this->setLogFileRoot("/var/log/"))
            {
                if (currentLevel_ <= LogLevel::DEBUG)
                {
                    std::string logMsg = "日志将写入: /var/log/" + this->generateLogFileName(this->app_name_);
                    logString(LogLevel::DEBUG, logMsg);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "创建默认日志文件时出错: " << e.what() << std::endl;
        }
    }

    Logger::~Logger()
    {
        if (logFile_.is_open())
        {
            logFile_.close();
        }
    }

    Logger &Logger::getInstance()
    {
        static Logger instance;
        return instance;
    }

    void Logger::setLevel(LogLevel level)
    {
        std::lock_guard<std::mutex> lock(logMutex_);
        currentLevel_ = level;
    }

    void Logger::setOutputType(LogOutputType type)
    {
        std::lock_guard<std::mutex> lock(logMutex_);
        outputTarget_ = type;
    }

    void Logger::setAppName(const std::string &appName)
    {
        std::lock_guard<std::mutex> lock(logMutex_);
        app_name_ = appName;
    }

    bool Logger::setLogFileRoot(const std::string &fileRoot)
    {
        std::lock_guard<std::mutex> lock(logMutex_);

        // 关闭已打开的文件
        if (logFile_.is_open())
        {
            logFile_.close();
        }

        // 打开新文件
        std::string filePath = (std::filesystem::path(fileRoot) / this->generateLogFileName(this->app_name_)).string();
        logFile_.open(filePath, std::ios::out | std::ios::app);
        if (!logFile_.is_open())
        {
            std::cerr << "Failed to open log file: " << filePath << std::endl;
            return false;
        }

        return true;
    }

    void Logger::showTimestamp(bool show)
    {
        std::lock_guard<std::mutex> lock(logMutex_);
        showTimestamp_ = show;
    }

    void Logger::showLogLevel(bool show)
    {
        std::lock_guard<std::mutex> lock(logMutex_);
        showLogLevel_ = show;
    }

    std::string Logger::getCurrentTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

        return ss.str();
    }

    std::string Logger::generateLogFileName(const std::string &prefix)
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << prefix << "-" << std::put_time(std::localtime(&time), "%Y%m%d-%H%M%S") << ".log";
        return ss.str();
    }

    std::string Logger::getLevelString(LogLevel level)
    {
        switch (level)
        {
        case LogLevel::TRACE:
            return "TRACE";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO ";
        case LogLevel::WARNING:
            return "WARN ";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::FATAL:
            return "FATAL";
        default:
            return "UNKN ";
        }
    }

    void Logger::writeLog(LogLevel level, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(logMutex_);

        std::stringstream logStream;

        // 添加时间戳
        if (showTimestamp_)
        {
            logStream << "[" << getCurrentTimestamp() << "] ";
        }

        // 添加日志级别
        if (showLogLevel_)
        {
            logStream << "[" << getLevelString(level) << "] ";
        }

        // 添加消息内容
        logStream << message;

        // 输出到控制台
        if (outputTarget_ == LogOutputType::CONSOLE || outputTarget_ == LogOutputType::BOTH)
        {
            if (level == LogLevel::ERROR || level == LogLevel::FATAL)
            {
                std::cerr << logStream.str() << std::endl;
            }
            else
            {
                std::cout << logStream.str() << std::endl;
            }
        }

        // 输出到文件
        if ((outputTarget_ == LogOutputType::FILE || outputTarget_ == LogOutputType::BOTH) && logFile_.is_open())
        {
            logFile_ << logStream.str() << std::endl;
            logFile_.flush();
        }
    }

} // namespace GryFlux
