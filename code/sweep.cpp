// Alpha x Beta x Gamma grid sweep for the geo/ego maze agent.
// Writes ONE csv per density:  <out>/<density>/grid_sweep.csv
//
// Build:
//   g++ -std=c++17 -O3 -ffast-math -march=native -pthread sweep.cpp cnpy.cpp -lz -o sweep
// Run:
//   ./sweep coarse                      # all 4 densities, coarse grid
//   ./sweep fine                        # all 4 densities, fine grid
//   ./sweep ultra                       # very fine grid (long run, see GridConfig below)
//   ./sweep coarse P0400                # only one density
//   ./sweep coarse --data ../data_jax_u8 P0100 P0400
//   ./sweep coarse --threads 1          # single-threaded
//
// Maze files must be uint8 (word_size = 1). Use the converted *_u8 files.

#include "cnpy.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <random>

namespace fs = std::filesystem;

static std::string fmt_time(double sec) {
    char b[64];
    long s = (long)sec;
    if (s >= 3600)      snprintf(b, sizeof b, "%ldh%02ldm", s / 3600, (s % 3600) / 60);
    else if (s >= 60)   snprintf(b, sizeof b, "%ldm%02lds", s / 60, s % 60);
    else                snprintf(b, sizeof b, "%lds", s);
    return b;
}

// ---------------------------------------------------------
// 1. Settings
// ---------------------------------------------------------
struct GridConfig {
    std::string name;
    double a_lo, a_hi, a_step;     // alpha range
    double b_lo, b_hi, b_step;     // beta range
    double g_step;                 // gamma 0..1
    int    agents_per_maze;        // trials per maze per (alpha,beta,gamma)
};

// Edit these freely.  alpha, beta in [0,5]; gamma in [0,1].
// NOTE: alpha and beta saturate quickly (the softmax is ~deterministic above ~3-4).
//                                   name      alpha: lo hi step    beta: lo hi step    gamma step  agents/maze
static const GridConfig COARSE = {"coarse", 0.0, 5.0, 0.5,    0.0, 5.0, 0.5,    0.1,   30};
static const GridConfig FINE   = {"fine",   0.0, 5.0, 0.1,    0.0, 5.0, 0.1,    0.02,  30};   // 51 x 51 x 51
static const GridConfig ULTRA  = {"ultra",  0.0, 5.0, 0.05,   0.0, 5.0, 0.05,   0.01,  30};   // 101 x 101 x 101

constexpr int MAX_STEPS = 200;      // horizon T. Blocked moves are rejected but still use up a step.
// Success is also recorded for these shorter horizons (from the same runs, no extra cost).
// All must be <= MAX_STEPS. Shortest possible path on a 16x16 grid is 30 steps.
constexpr int NH = 3;
constexpr int HORIZONS[NH] = {50, 100, 150};
static_assert(HORIZONS[NH - 1] <= MAX_STEPS, "horizons must not exceed MAX_STEPS");
static const std::string DEFAULT_DATA_DIR = "../../data_jax_u8";
static const std::vector<std::string> ALL_DENSITIES = {"P0100", "P0200", "P0300", "P0400"};

// ---------------------------------------------------------
// 2. Fast RNG (xoshiro256**), one instance per thread
// ---------------------------------------------------------
struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    explicit Rng(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
};

// ---------------------------------------------------------
// 3. Policy tables (same maths as before, computed once per alpha/beta)
//    Action order: 0=North(-y) 1=South(+y) 2=East(+x) 3=West(-x)
// ---------------------------------------------------------
static const int DX[4] = {0, 0, 1, -1};
static const int DY[4] = {-1, 1, 0, 0};

static std::array<double, 4> softmax4(const std::array<double, 4>& sc) {
    double mx = std::max(std::max(sc[0], sc[1]), std::max(sc[2], sc[3]));
    std::array<double, 4> p;
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) { p[i] = std::exp(sc[i] - mx); sum += p[i]; }
    for (int i = 0; i < 4; ++i) p[i] /= sum;
    return p;
}

// geo[cell*4 + a] : goal-direction softmax, depends only on cell and beta
static std::vector<double> build_geo(int rows, int cols, double beta) {
    std::vector<double> geo(rows * cols * 4);
    const int gx = cols - 1, gy = rows - 1;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double dx = gx - x, dy = gy - y;
            double mag = std::sqrt(dx * dx + dy * dy);
            std::array<double, 4> p;
            if (mag < 1e-9) {
                p = {0.25, 0.25, 0.25, 0.25};
            } else {
                double ux = dx / mag, uy = dy / mag;
                std::array<double, 4> sc;
                for (int a = 0; a < 4; ++a) sc[a] = beta * (DX[a] * ux + DY[a] * uy);
                p = softmax4(sc);
            }
            for (int a = 0; a < 4; ++a) geo[(y * cols + x) * 4 + a] = p[a];
        }
    }
    return geo;
}

// ego[mask][a] : mask bit a = 1 if moving in direction a is blocked (wall or boundary)
static std::array<std::array<double, 4>, 16> build_ego(double alpha) {
    std::array<std::array<double, 4>, 16> ego;
    for (int m = 0; m < 16; ++m) {
        std::array<double, 4> sc;
        for (int a = 0; a < 4; ++a) sc[a] = -alpha * ((m >> a) & 1);
        ego[m] = softmax4(sc);
    }
    return ego;
}

// ---------------------------------------------------------
// 4. Simulation of all mazes x agents for one (alpha,beta,gamma)
// ---------------------------------------------------------
struct Agg { long long runs = 0, succ = 0, sumT = 0, capped = 0; long long by_h[NH] = {}; };

static Agg run_combo(const std::vector<uint8_t>& mask, int num_mazes, int rows, int cols,
                     const std::vector<double>& geo,
                     const std::array<std::array<double, 4>, 16>& ego,
                     double gamma, int agents, int max_steps, Rng& rng) {
    Agg agg;
    const int ncell = rows * cols;
    const int goal = ncell - 1;
    const int step[4] = {-cols, cols, 1, -1};
    const double om = 1.0 - gamma;

    for (int m = 0; m < num_mazes; ++m) {
        const uint8_t* mk = &mask[(size_t)m * ncell];
        for (int a = 0; a < agents; ++a) {
            int pos = 0;
            int t = 0;
            bool ok = false;
            for (t = 1; t <= max_steps; ++t) {
                const double* g = &geo[pos * 4];
                const std::array<double, 4>& e = ego[mk[pos]];
                double c0 = gamma * g[0] + om * e[0];
                double c1 = c0 + gamma * g[1] + om * e[1];
                double c2 = c1 + gamma * g[2] + om * e[2];
                double u = rng.uniform();
                int act = (u < c0) ? 0 : (u < c1) ? 1 : (u < c2) ? 2 : 3;
                if (!((mk[pos] >> act) & 1)) pos += step[act];   // blocked -> stay, step still consumed
                if (pos == goal) { ok = true; break; }
            }
            agg.runs++;
            if (ok) {
                agg.succ++; agg.sumT += t; agg.capped += t;
                for (int h = 0; h < NH; ++h) if (t <= HORIZONS[h]) agg.by_h[h]++;
            }
            else    { agg.capped += max_steps; }
        }
    }
    return agg;
}

// ---------------------------------------------------------
// 5. Helpers
// ---------------------------------------------------------
static std::vector<double> make_range(double lo, double hi, double step) {
    int n = (int)std::lround((hi - lo) / step) + 1;
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = lo + i * step;   // no float accumulation
    return v;
}

// bit a set  <=>  moving in direction a from this cell is blocked
static std::vector<uint8_t> build_masks(const uint8_t* data, int num_mazes, int rows, int cols) {
    std::vector<uint8_t> mask((size_t)num_mazes * rows * cols, 0);
    for (int m = 0; m < num_mazes; ++m) {
        const uint8_t* mz = data + (size_t)m * rows * cols;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                uint8_t b = 0;
                for (int a = 0; a < 4; ++a) {
                    int nx = x + DX[a], ny = y + DY[a];
                    bool blk = (nx < 0 || nx >= cols || ny < 0 || ny >= rows) ||
                               mz[ny * cols + nx] == 1;
                    if (blk) b |= (1 << a);
                }
                mask[(size_t)m * rows * cols + y * cols + x] = b;
            }
        }
    }
    return mask;
}

// ---------------------------------------------------------
// 6. One density
// ---------------------------------------------------------
static bool process_density(const std::string& density, const std::string& data_dir,
                            const GridConfig& cfg, const std::string& out_root, uint64_t base_seed,
                            unsigned req_threads) {
    std::string filepath = data_dir + "/N16_" + density + "_test_solvable_random.npy";
    cnpy::NpyArray arr;
    try { arr = cnpy::npy_load(filepath); }
    catch (const std::exception& e) { std::cerr << "Cannot load " << filepath << ": " << e.what() << "\n"; return false; }

    // ---- safety checks: this is exactly what went wrong before ----
    if (arr.word_size != 1) { std::cerr << filepath << ": word_size=" << arr.word_size << " (need uint8 / 1). Convert the file first.\n"; return false; }
    if (arr.shape.size() != 3) { std::cerr << filepath << ": expected 3-D array\n"; return false; }
    if (arr.fortran_order) { std::cerr << filepath << ": fortran_order arrays not supported\n"; return false; }
    const uint8_t* data = arr.data<uint8_t>();
    const int num_mazes = (int)arr.shape[0], rows = (int)arr.shape[1], cols = (int)arr.shape[2];
    size_t walls = 0;
    for (size_t i = 0; i < arr.num_vals; ++i) {
        if (data[i] > 1) { std::cerr << filepath << ": found value " << (int)data[i] << " (expected only 0/1)\n"; return false; }
        walls += data[i];
    }
    int bad = 0;
    for (int m = 0; m < num_mazes; ++m) {
        const uint8_t* mz = data + (size_t)m * rows * cols;
        if (mz[0] == 1 || mz[rows * cols - 1] == 1) ++bad;
    }
    std::cout << "\n=== " << density << " | " << num_mazes << " mazes " << rows << "x" << cols
              << " | wall fraction " << std::fixed << std::setprecision(4) << (double)walls / arr.num_vals << " ===\n";
    if (bad) std::cout << "  WARNING: " << bad << " mazes have a blocked start or goal cell\n";

    std::vector<uint8_t> mask = build_masks(data, num_mazes, rows, cols);

    const std::vector<double> alphas = make_range(cfg.a_lo, cfg.a_hi, cfg.a_step);
    const std::vector<double> betas  = make_range(cfg.b_lo, cfg.b_hi, cfg.b_step);
    const std::vector<double> gammas = make_range(0.0, 1.0, cfg.g_step);
    const int na = (int)alphas.size(), nb = (int)betas.size(), ng = (int)gammas.size();
    const int npairs = na * nb;
    std::vector<Agg> results((size_t)npairs * ng);

    std::cout << "  grid: " << na << " alpha x " << nb << " beta x " << ng << " gamma = "
              << (size_t)npairs * ng << " combos, " << (long long)num_mazes * cfg.agents_per_maze
              << " runs each, max_steps=" << MAX_STEPS << "\n";

    unsigned nthreads = req_threads ? req_threads : std::max(1u, std::thread::hardware_concurrency());
    nthreads = std::min<unsigned>(nthreads, npairs);
    std::atomic<int> next_pair{0}, done_pairs{0};
    std::mutex print_mx;
    int last_pct = -1;
    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            int idx = next_pair.fetch_add(1);
            if (idx >= npairs) break;
            int ia = idx / nb, ib = idx % nb;
            Rng rng(base_seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(idx + 1));
            std::vector<double> geo = build_geo(rows, cols, betas[ib]);
            auto ego = build_ego(alphas[ia]);
            for (int ig = 0; ig < ng; ++ig) {
                results[(size_t)idx * ng + ig] =
                    run_combo(mask, num_mazes, rows, cols, geo, ego, gammas[ig],
                              cfg.agents_per_maze, MAX_STEPS, rng);
            }
            int d = done_pairs.fetch_add(1) + 1;
            int pct = d * 100 / npairs;
            std::lock_guard<std::mutex> lk(print_mx);
            if (pct != last_pct) {
                last_pct = pct;
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double eta = (pct > 0) ? el * (100 - pct) / pct : 0.0;
                std::cout << "\r  progress " << std::setw(3) << pct << "%  elapsed " << fmt_time(el)
                          << "  ETA " << fmt_time(eta) << "      " << std::flush;
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n  done in " << std::setprecision(1) << total << "s using " << nthreads << " threads\n";

    // ---- write CSV ----
    fs::path dir = fs::path(out_root) / density;
    fs::create_directories(dir);
    std::ofstream f(dir / "grid_sweep.csv");
    f << "alpha,beta,gamma,runs,successes,success_rate,mfpt_success,mfpt_capped";
    for (int h = 0; h < NH; ++h) f << ",success_T" << HORIZONS[h];
    f << "\n";
    f << std::fixed;
    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ig = 0; ig < ng; ++ig) {
                const Agg& r = results[((size_t)ia * nb + ib) * ng + ig];
                double sr = (double)r.succ / r.runs;
                double mf = r.succ ? (double)r.sumT / r.succ : -1.0;
                double cp = (double)r.capped / r.runs;
                f << std::setprecision(4) << alphas[ia] << "," << betas[ib] << "," << gammas[ig] << ","
                  << r.runs << "," << r.succ << "," << std::setprecision(6) << sr << ","
                  << std::setprecision(3) << mf << "," << cp << std::setprecision(6);
                for (int h = 0; h < NH; ++h) f << "," << (double)r.by_h[h] / r.runs;
                f << "\n";
            }
        }
    }
    std::ofstream meta(dir / "meta.txt");
    meta << "mode=" << cfg.name << "\ndensity=" << density << "\nsource=" << filepath
         << "\nmazes=" << num_mazes << "\nagents_per_maze=" << cfg.agents_per_maze
         << "\nmax_steps=" << MAX_STEPS << "\nseed=" << base_seed
         << "\nwall_fraction=" << (double)walls / arr.num_vals << "\n";
    std::cout << "  wrote " << (dir / "grid_sweep.csv").string() << "\n";
    return true;
}

// ---------------------------------------------------------
// 7. Main
// ---------------------------------------------------------
int main(int argc, char** argv) {
    const std::string mode = (argc >= 2) ? argv[1] : "";
    const GridConfig* cfgp = (mode == "coarse") ? &COARSE : (mode == "fine") ? &FINE
                           : (mode == "ultra")  ? &ULTRA  : nullptr;
    if (!cfgp) {
        std::cerr << "usage: " << argv[0] << " coarse|fine|ultra [--data DIR] [--threads N] [P0100 P0200 ...]\n";
        return 1;
    }
    const GridConfig& cfg = *cfgp;
    std::string data_dir = DEFAULT_DATA_DIR;
    std::vector<std::string> densities;
    unsigned req_threads = 0;   // 0 = use all cores; --threads 1 = single-threaded
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (a == "--threads" && i + 1 < argc) req_threads = (unsigned)std::stoi(argv[++i]);
        else densities.push_back(a);
    }
    if (densities.empty()) densities = ALL_DENSITIES;

    std::random_device rd;
    uint64_t base_seed = ((uint64_t)rd() << 32) ^ rd();
    const std::string out_root = "results_" + cfg.name;
    std::cout << "mode=" << cfg.name << "  out=" << out_root << "/<density>/grid_sweep.csv  seed=" << base_seed << "\n";

    for (const auto& d : densities)
        if (!process_density(d, data_dir, cfg, out_root, base_seed, req_threads)) return 1;
    return 0;
}