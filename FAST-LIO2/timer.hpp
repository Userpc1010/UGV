#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <inttypes.h>

class Timer
{
public:

  void setperiod(unsigned long period);
  void every(unsigned long period, void (*callback)(void),bool start);

  void stop();
  void start();
  void update();

  unsigned int millis();

  uint32_t Time_milliseconds();
  uint64_t Time_microseconds();
  double Time_microseconds_d();


private:
  unsigned long period;
  unsigned long lastEventTime;
  void (*callback)(void);
  bool is_working;

  uint64_t epochMilli,epochMicro;

};
#endif // TIMER_H
