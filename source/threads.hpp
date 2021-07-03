// Information: parallelisation with access to thread ID and indices

#pragma once

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

class Tp {

public:

    // ----- CONSTRUCTOR AND DESTRUCTOR -----
    Tp(int nthreads) :
        threads(),
        nthreads(nthreads),
        cap(0),
        ret(0) {
        for (int t = 0; t < nthreads; t++) {
            threads.push_back(std::thread(&Tp::loop, this));
        }
        wait();
    }

    ~Tp() {
        call = nullptr;
        args = nullptr;

        c.notify_all();
        for (int t = 0; t < nthreads; t++) {
            threads[t].join();
        }
    }

    // ----- PUBLIC METHODS -----
    template <class L>
    void parallel(int tasks, const L& immu) {
        call = &Tp::wrap<L>;
        args = &immu;

        cap = tasks;
        ret = 0;
        up  = 0;

        c.notify_all();
        wait();
    }

    int size() {
        return nthreads;
    }

protected:

    // ----- PROTECTED METHODS -----


    void (Tp::*call)(int t);

    template <class L>
    void wrap(int t) {
        int delta = (cap + nthreads - 1) / nthreads;
        int from  = t * delta;
        int to    = (t + 1) * delta;

        for (int it = from; it < std::min(cap, to); it++) {
            (*reinterpret_cast<const L*>(args))(it, t);
        }
    }

    void loop() {
        int t;
        {
            std::lock_guard<std::mutex> lock(m);
            t = cap++;
        }

        while (true) {
            {
                std::unique_lock<std::mutex> lock(m);
                ret++;
                if (ret == nthreads) {
                    w.notify_all();
                }
                c.wait(lock);
            }

            if (this->call) {
                (this->*call)(t);
            } else {
                break;
            }
        }
    }

    void wait() {
        std::unique_lock<std::mutex> lock(m);
        while (ret < nthreads) {
            w.wait(lock);
        }
    }

    // ----- PROTECTED VARIABLES -----
    const void               *args;
    std::vector<std::thread> threads;
    int                      nthreads;
    int                      cap;
    int                      up;
    int                      ret;
    std::mutex               m;
    std::condition_variable  c;
    std::condition_variable  w;
};
