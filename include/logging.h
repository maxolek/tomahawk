#pragma once
#include <string>
#include "helpers.h"
#include "session.h"
#include <filesystem>

namespace fs = std::filesystem;

struct RunContext {
    bool is_new_game = false; // flag to set certain log attributes when a new game is triggered
    std::string game_uuid; // updates when a new game starts
    std::string search_uuid; // updates every search
};
inline RunContext g_run_context{};

struct Logging {
    // ---- toggles ----
    static inline bool track_uci           = false;

    // ---- directory ----
    // uci is .log, rest are .jsonl
    static inline const fs::path project_root = PROJECT_ROOT; // compile time constant (str)
    static inline const fs::path san_jacinto_logs = project_root / "../san-jacinto/logs";
    static inline fs::path DEFAULT_LOG_DIR = san_jacinto_logs / "test_logs";
    static inline fs::path log_dir = DEFAULT_LOG_DIR;
    static inline std::ofstream uci_file;
    static inline std::ofstream search_file;
    static inline std::ofstream game_file;
    static inline std::ofstream timing_file;

    static fs::path log_file_name(const std::string& name) {
        fs::path file(name);
        std::string stem = file.stem().string();
        std::string ext = file.extension().string();
        return log_dir / fs::path(stem + "_" + std::to_string(instanceID()) + ext);
    }

    // life cycle
    static void initFiles() {
        std::filesystem::create_directories(log_dir);

        if (track_uci)
            uci_file.open(log_file_name("uci.log"), std::ios::app);
        // search, game, and timing file handles are managed by the modules that write them.
    }

    static void closeFiles() {
        if (uci_file.is_open())     uci_file.close();
        if (search_file.is_open())  search_file.close();
        if (game_file.is_open())    game_file.close();
        if (timing_file.is_open())  timing_file.close();
    }

    static void reopenFiles() {
        closeFiles();
        initFiles();
    }

    // uci handlers
    static inline void logUCIin(const std::string& line) {
        if (!Logging::track_uci) return;
        Logging::uci_file << "< " << line << "\n";
    }
    static inline void logUCIout(const std::string& line) {
        if (!Logging::track_uci) return;
        Logging::uci_file << " > " << line << "\n";
    }

    // ---- setters (called from setoption) ----
    static void setLogDir(const std::string& path_str) {
        setLogDir(std::filesystem::path(path_str));
    }
    static void setLogDir(const fs::path& path) {
        fs::path dir = path;

        if (dir.is_relative())
            dir = fs::absolute(dir);

        if (log_dir == dir) return;

        log_dir = dir;
        initFiles();
    }

    static void setTrackUCI(bool v) {
        if (track_uci == v) return;
        track_uci = v;
        reopenFiles();
    }
};