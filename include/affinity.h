// affinity.h
// Which logical CPUs the search threads run on (UCI option "Affinity").
//
//   auto  (default) performance cores first. With one search thread, the thread
//         may run on any logical CPU of the performance cores (the OS picks), so
//         several single-threaded engine instances, e.g. concurrent test games,
//         spread over them. With more threads, thread i is pinned to the i-th
//         CPU of Topology::order: one logical CPU per physical core, fastest
//         cores first, efficiency cores interleaved over their shared L2
//         clusters, and the second logical CPU of SMT cores last.
//   none  threads are not pinned: each may run on every CPU of the process (on
//         Windows with several processor groups, on one whole group).
//   list  e.g. "0,2,4,6" or "0-3,8": thread i is pinned to the i-th listed CPU.
//         Each CPU may be listed once.
//
// There are never more search threads than CPUs to run them (cpu_limit): the
// engine caps Threads at the listed CPUs, or else at the logical CPUs of the
// process, so no two search threads share a CPU.
//
// CPU numbers are the operating system's logical processor numbers (as in Task
// Manager); on Windows with several processor groups, cpu = 64 * group + number.
// Only the CPUs the process was given are used, re-read at every search: the
// affinity mask and, on Windows 10+, the process default CPU set (fastchess
// -use-affinity uses CPU sets on Windows 11); on Linux, the affinity of the
// initial thread (taskset, match runners). So an assignment made by the program
// that started the engine is respected. Each search thread applies its placement
// to itself at the start of every search. Supported on Windows and Linux;
// elsewhere nothing is pinned.

#ifndef AFFINITY_H
#define AFFINITY_H

#include <string>
#include <vector>

namespace affinity {

struct Topology {
    std::vector<int> order;       // every logical CPU, in the order `auto` places threads
    std::vector<int> performance; // logical CPUs of the performance cores, ascending
    int performance_cores = 0;    // physical cores within 10% of the fastest core type
    int other_cores = 0;          // slower physical cores (efficiency cores)
    bool smt = false;             // some core has more than one logical CPU
    bool supported = false;       // CPU placement works on this platform
};

// Detected on first use.
const Topology& topology();
// One line for "info string", e.g. "16 logical cpus: 4 performance + 8 efficiency cores ...".
std::string describe(const Topology& topo);

// A parsed Affinity option value.
struct Spec {
    enum class Mode { Auto, None, List };
    Mode mode = Mode::Auto;
    std::vector<int> cpus; // Mode::List, in thread order
};

// Parses an option value (case-insensitive; empty means auto). On failure returns
// false, leaves `spec` unchanged and explains in `error`.
bool parse(const std::string& text, Spec& spec, std::string& error);
// Canonical option text: "auto", "none" or the CPU list.
std::string to_string(const Spec& spec);

// Placement of the threads of one search.
struct Plan {
    std::vector<std::vector<int>> threads;  // CPUs each thread runs on; empty = leave it alone
    std::vector<std::vector<int>> unpinned; // each thread's CPUs when not pinned (also the
                                            // fallback if the OS refuses `threads[i]`)
    std::string summary;                    // one line for "info string"
};

// Placement of `thread_count` search threads (index 0 = the main thread), limited
// to the CPUs the process may use right now.
Plan make_plan(const Spec& spec, int thread_count);

// The most search threads that can run under `spec` without sharing a CPU: the
// listed CPUs available to the process, or else all logical CPUs it may use.
// Recomputed on every call, since the process affinity can change.
int cpu_limit(const Spec& spec);

// Restricts the calling thread to `cpus` (nothing to do for an empty list).
// Returns false if the operating system refused.
bool apply_to_current_thread(const std::vector<int>& cpus);

// "0-3,8,6": the numbers in their order, runs of three or more as ranges.
std::string format_cpus(const std::vector<int>& cpus);

} // namespace affinity

#endif // AFFINITY_H
