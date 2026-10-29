// uci.h
#ifndef UCI_H
#define UCI_H

#include "engine.h"
#include "session.h"
#include <string>
#include <sstream>
#include <iostream>
#include <atomic>

class UCI {
public:
    UCI(Engine& engine);
    ~UCI();

    void loop();
    void runSearch(bool evalRequest);

    void handleCommand(const std::string& line);
    void handlePosition(std::istringstream& iss);
    void handleGo(std::istringstream& iss);
    void handleSetOption(std::istringstream& iss);

    void handleZobrist();
    void handleClearTT();
    void handleConfigList(fs::path dir);
    void handleConfigSet(std::string name);

private:
    Engine* engine;
    std::thread search_thread;
    std::atomic<bool> search_active{false};

    bool sendEval = false;
    int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = -1, depth = -1;
};

#endif
