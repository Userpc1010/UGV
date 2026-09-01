#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <chrono>
#include <ctime>

#include "timer.h"

void Timer::every(unsigned long period, void (*callback)(),bool start)
{
    struct timeval tv;
    gettimeofday (&tv, NULL) ;
    this->epochMilli = (uint64_t)tv.tv_sec * (uint64_t)1000    + (uint64_t)(tv.tv_usec / 1000) ;
    this->epochMicro = (uint64_t)tv.tv_sec * (uint64_t)1000000 + (uint64_t)(tv.tv_usec) ;

    this->period = period;
    this->callback = callback;
    this->lastEventTime = millis();
    this->is_working = start;

return;
}

void Timer::setperiod(unsigned long period)
{
    this->period = period;
return;
}

void Timer::stop()
{
    this->is_working = false;
return;
}

void Timer::start()
{
    this->is_working = true;
return;
}

unsigned int Timer::millis()
{
  uint64_t now;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  now  = (uint64_t)tv.tv_sec * (uint64_t)1000 + (uint64_t)(tv.tv_usec / 1000);
return (uint32_t)(now - this->epochMilli);
}

uint32_t Timer::Time_milliseconds()
{
  struct timeval t;
  gettimeofday(&t, NULL); // get current time
  uint32_t milliseconds = t.tv_sec*1000LL + t.tv_usec/1000; // calculate milliseconds
  return milliseconds;
}

uint64_t Timer::Time_microseconds()
{
  struct timeval t;
  gettimeofday(&t, NULL); // get current time
  uint64_t microseconds = t.tv_sec*1000000ULL + t.tv_usec; // calculate microseconds
  return microseconds;
}

double Timer::Time_microseconds_d()
{
 struct timeval t;
 gettimeofday(&t, NULL); // get current time
 uint64_t microseconds = t.tv_sec*1000000ULL + t.tv_usec; // calculate microseconds
 return microseconds / 1000000.0;
}

void Timer::update()
{
    if(this->is_working == true)
    {
        unsigned long now = millis();
        if (now - this->lastEventTime >= this->period)
        {
             //printf("Period: %ld millis\n", this->period);  // uncomment for test!!
             //printf("Time: %ld millis\n", now - this->lastEventTime); // uncomment for test!!
            (*callback)();
            this->lastEventTime = now;
        }
    }
}
