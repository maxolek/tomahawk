#include <UCI.h>
#include <session.h>
#include <engine.h>
#include <board.h>
#include <precomp.h>
#include <iostream>
#include <atomic>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <sched.h>
    #include <pthread.h>
    #ifdef __APPLE__
        #include <mach/mach.h>
        #include <mach/thread_policy.h>

    #endif
#endif

// version name set at compile time
#ifndef ENGINE_ID
#define ENGINE_ID "unknown"
#endif

// ---------------------
// Thread scheduling
// ---------------------
// search threads are placed on CPUs at the start of every search
// according to the affinity option (affinity.h)
// nothing is pinned at startup 
void set_thread_precendence() {
#if defined(__APPLE__)
    thread_port_t thread = mach_thread_self();
    thread_precedence_policy_data_t policy = { 63 }; // highest precedence
    thread_policy_set(
        thread, 
        THREAD_PRECEDENCE_POLICY, 
        (thread_policy_t)&policy,
        THREAD_PRECEDENCE_POLICY_COUNT
    );
#endif
}

// ---------------------
// UCI listening loop
// ---------------------
std::atomic<bool> keepListening(true);

void listenLoop(UCI& uci) {
    std::string line;
    while (keepListening) {
        if (!std::getline(std::cin, line)) {
            keepListening = false;
            break;
        }
        uci.handleCommand(line);
        if (line == "quit") {
            keepListening = false;
            break;
        }
    }
}

// ---------------------
// main
// ---------------------
int main() {
    // environment
    set_thread_precendence();
    initInstance();      // PID-based instance id
    startNewSession();   // session=1

    // inits
    Logging::initFiles();
    PrecomputedMoveData::init();
#ifdef _WIN32
    Magics::initPEXTMagics(); //initMagics();
#else 
    Magics::initMagics();
#endif

    // engine
    static Engine engine;
    UCI uci(engine);

    // uci loop
    // searches run on this listener thread, never on the initial thread
    // on linux the initial thread's affinity is read as "the CPUs the process was given" (affinity.cpp)
    std::thread listener([&uci](){
        uci.loop(); // loop internally
    });
    listener.join();

    return 0;
}
