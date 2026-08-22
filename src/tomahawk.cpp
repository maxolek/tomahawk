#include <UCI.h>
#include <session.h>
#include <engine.h>
#include <board.h>
#include <PrecomputedMoveData.h>
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
// CPU affinity
// ---------------------
void pin_to_pcores() {
#ifdef _WIN32
    // P-cores (0,1,6,7,8,9,18,19)
    // star cores (0,6,8)
    DWORD_PTR mask = 0;
    //mask |= (1ULL << 0) | (1ULL << 6) | (1ULL << 8); // star cores
    mask |= (1ULL << 0) | (1ULL << 1) | (1ULL << 6) | (1ULL << 7) | (1ULL << 8) | (1ULL << 9) | (1ULL << 18) | (1ULL << 19); // performance cores
    HANDLE hProc = GetCurrentProcess();
    if (!SetProcessAffinityMask(hProc, mask)) 
        std::cerr << "Failed to set CPU affinity, error " << GetLastError() << "\n";
#elif defined(__APPLE__)
    thread_port_t thread = mach_thread_self();
    thread_precedence_policy_data_t policy = {63};
    thread_policy_set(thread,
                    THREAD_PRECEDENCE_POLICY,
                    (thread_policy_t)&policy,
                    THREAD_PRECEDENCE_POLICY_COUNT);
#else
    // Linux - no-op or use sched_setaffinity if needed
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
    pin_to_pcores();
    initInstance();      // PID-based instance id
    startNewSession();   // session=1

    // inits
    Logging::initFiles();
    PrecomputedMoveData::init();
    Magics::initPEXTMagics(); //initMagics();

    // engine
    Engine engine;
    UCI uci(engine);

    // uci loop
    std::thread listener([&uci](){
        uci.loop(); // loop internally
    });
    listener.join();

    return 0;
}
