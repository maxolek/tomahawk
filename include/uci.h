// uci.h
#ifndef UCI_H
#define UCI_H

#include "engine.h"
#include "session.h"
#include <string>
#include <sstream>
#include <iostream>
#include <atomic>
#include <condition_variable>
#include <mutex>

class UCI {
public:
    UCI(Engine& engine);
    ~UCI();

    void loop();
    void runSearch(bool evalRequest);
    void waitForSearch();

    void handleCommand(const std::string& line);
    void handlePosition(std::istringstream& iss);
    void handleGo(std::istringstream& iss);
    void handleSetOption(std::istringstream& iss);

    void handleZobrist();
    void handleClearTT();

private:
    void searchLoop();
    void updateSearchThreadAffinity();

    Engine* engine;
    std::thread search_thread;
    std::atomic<bool> search_active{false};
    std::mutex search_mutex;
    std::condition_variable search_cv;
    bool search_pending = false;
    bool affinity_pending = false;
    bool shutdown_search_worker = false;
    bool worker_ready = false;
    bool pending_eval_request = false;

    bool sendEval = false;
    int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = -1, depth = -1;
};

#endif
