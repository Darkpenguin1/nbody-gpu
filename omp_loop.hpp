#ifndef __OMPLOOP__
#define __OMPLOOP__

#include <functional>

// You need to add -fopenmp to CXXFLAGS and LDFLAGS for this to work.
// set number of thread by called setNbThread.
class OmpLoop {

  int nbthread;
  int granularity;

public:

  OmpLoop()
    :nbthread(1), granularity(1)
  {}

  /// @brief selects the number of thread that will be used in the
  /// parfor construct. This value must be positive.
  void setNbThread(int t) {
    nbthread = t;
  }

  /// @brief selects the granularity that will be used in the parfor
  /// construct.
  ///
  /// In other words, iterations of the parallel for will be
  /// distributed in batches of "granularity" iterations.  In general,
  /// a lower granularity will enable a better load balance, and so
  /// more parallelism. But a smaller value will also increase
  /// scheduling overheads.
  void setGranularity(size_t g) {
    granularity = g;
  }

  /// @brief execute the function f multiple times with different
  /// parameters possibly in parallel
  ///
  /// f will be executed multiple times with parameters starting at
  /// beg, no greater than end, in inc increment. These execution may
  /// be in parallel
  void parfor (size_t beg, size_t end, size_t increment,
               std::function<void(size_t)> f
               ) {
#pragma omp parallel num_threads(nbthread)
    {
#pragma omp for schedule(dynamic, granularity) 
      for (size_t i=beg; i<end; i+= increment) {
	f(i);
      }
    }
  }

  /// @brief execute the function f multiple times with different
  /// parameters possibly in parallel
  ///
  /// f will be executed multiple times with parameters starting at
  /// beg, no greater than end, in inc increment. These execution may
  /// be in parallel.
  ///
  /// Each thread that participate in the calculation will have its
  /// own TLS object.
  ///
  /// Each thread will execute function before prior to any
  /// calculation on a TLS object.
  ///
  /// Each thread will be passed the TLS object it executed before on.
  ///
  /// Once the iterations are complete, each thread will execute after
  /// on the TLS object. No two thread can execute after at the same time.
  template<typename TLS>
  void parfor (size_t beg, size_t end, size_t increment,
               std::function<void(TLS&)> before,
               std::function<void(size_t, TLS&)> f,
               std::function<void(TLS&)> after
               ) {
#pragma omp parallel num_threads(nbthread)
    {
      TLS tls;
      before(tls);
      
#pragma omp for schedule(dynamic, granularity) 
      for (size_t i=beg; i<end; i+= increment) {
	f(i, tls);
      }
#pragma omp critical
      after(tls);
    }
  }

};

#endif
