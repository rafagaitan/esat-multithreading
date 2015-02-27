#pragma once


#include <iostream>
#include <string>

#ifndef __APPLE__
#include <chrono>
struct ScopedTimer
{
    typedef std::chrono::duration<float> float_seconds;
    ScopedTimer(const std::string& infoText, unsigned int numTests = 1) :
        _infoText(infoText),
        _start(std::chrono::system_clock::now()),
        _numTests(numTests)
    { }
    ~ScopedTimer()
    {
        auto elapsed = std::chrono::duration_cast<float_seconds>(std::chrono::system_clock::now() - _start);
        std::cout << "Elapsed time for " << _infoText << ": " << (float)elapsed.count() / (float)_numTests << std::endl;
    }
    std::string                                        _infoText;
    std::chrono::time_point<std::chrono::system_clock> _start;
    unsigned int _numTests;
};
#else
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>
struct ScopedTimer
{
    typedef uint64_t Timestamp_t;

    std::string  _infoText;
    Timestamp_t  _start;
    unsigned int _numTests;
    double       _secsPerTick;

    #if defined(_POSIX_TIMERS) && ( _POSIX_TIMERS > 0 ) && defined(_POSIX_MONOTONIC_CLOCK)
        #include <time.h>

        Timestamp_t tick()
        {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            return ((Timestamp_t)ts.tv_sec)*1000000+(Timestamp_t)ts.tv_nsec/1000;
        }
    #else
        #include <sys/time.h>

        Timestamp_t tick()
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            return ((Timestamp_t)tv.tv_sec)*1000000+(Timestamp_t)tv.tv_usec;
        }
    #endif
    ScopedTimer(const std::string& infoText, unsigned int numTests = 1) :
        _infoText(infoText),
        _start(tick()),
        _numTests(numTests),
	_secsPerTick(1.0/(double)1000000)
    { }
    ~ScopedTimer()
    {
        double elapsed = (double)(tick() - _start)*_secsPerTick;
        std::cout << "Elapsed time for " << _infoText << ": " << elapsed / (double)_numTests << std::endl;
    }

};

#endif

