#include <uci.h>
#include <session.h>
#include <engine.h>
#include <board.h>
#include <precomp.h>

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

    // Searches run on a worker thread while this loop continues reading commands.
    uci.loop();

    return 0;
}
