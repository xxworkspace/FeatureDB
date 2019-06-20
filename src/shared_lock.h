
#include <atomic>
#include <mutex>
#include <emmintrin.h>

class SharedLock{
private:
  bool wflag;
  std::mutex mtx;
  std::atomic<uint32_t> readers;
public:
  SharedLock():
    readers(0),wflag(false){}

  void rLock(){
    if(wflag){
      mtx.lock();
      mtx.unlock();
    }
    readers ++;
  }

  void rRelease(){
    readers --;
  }

  void wLock(){
    mtx.lock();
    wflag = true;
    while(readers > 0) _mm_pause();
  }

  void wRelease(){
    wflag = false;
    mtx.unlock();
  }
};
