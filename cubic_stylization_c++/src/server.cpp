// cubify_server: the C++ solver process the Blender add-on talks to.
//
// Blender starts one server per session and keeps it alive; meshes are sent
// over stdin/stdout (see protocol.h) and each gets a server-side session that
// holds the prefactorized system, so interactive drags only re-solve.
//
//   cubify_server              serve requests on stdin/stdout
//   cubify_server --version    print the version and exit
//   cubify_server --selftest   cubify a generated sphere and print timings

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "cubic_stylizer.h"
#include "protocol.h"
#include "thin_walls.h"

namespace {

using namespace cubify;

#define CUBIFY_STR2(x) #x
#define CUBIFY_STR(x) CUBIFY_STR2(x)
const char* kServerVersion = "cubify_server 1.0.0 (Eigen " CUBIFY_STR(EIGEN_WORLD_VERSION) "." CUBIFY_STR(
    EIGEN_MAJOR_VERSION) "." CUBIFY_STR(EIGEN_MINOR_VERSION) ")";

// ---------------- I/O ----------------

bool read_exact(void* dst, size_t n) {
  return std::fread(dst, 1, n, stdin) == n;
}

void write_all(const void* src, size_t n) {
  if (n > 0 && std::fwrite(src, 1, n, stdout) != n) throw std::runtime_error("stdout closed");
}

void write_header(uint32_t code, uint64_t size) {
  uint8_t h[protocol::kHeaderSize] = {};
  std::memcpy(h, &code, 4);
  std::memcpy(h + 8, &size, 8);
  write_all(h, sizeof h);
}

// Accumulates a response payload.
class Writer {
 public:
  template <class T>
  void put(const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  void put_bytes(const void* src, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    buf_.insert(buf_.end(), p, p + n);
  }
  void send(uint32_t code = protocol::kOk) {
    write_header(code, buf_.size());
    write_all(buf_.data(), buf_.size());
    std::fflush(stdout);
  }

 private:
  std::vector<uint8_t> buf_;
};

void send_error(const std::string& msg) {
  Writer w;
  w.put_bytes(msg.data(), msg.size());
  w.send(protocol::kError);
}

// Bounds-checked reader over a request payload. Arrays are memcpy'd out, so
// payload alignment does not matter.
class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), left_(n) {}
  template <class T>
  T get() {
    T v;
    take(&v, sizeof(T));
    return v;
  }
  void take(void* dst, size_t n) {
    if (n > left_) throw std::invalid_argument("truncated request");
    if (n > 0) std::memcpy(dst, p_, n);
    p_ += n;
    left_ -= n;
  }
  RowMatX3d mat3d(int32_t rows) {
    RowMatX3d M(rows, 3);
    take(M.data(), sizeof(double) * 3 * static_cast<size_t>(rows));
    return M;
  }

 private:
  const uint8_t* p_;
  size_t left_;
};

// ---------------- sessions ----------------

struct Session {
  std::unique_ptr<CubicStylizer> stylizer;
  int threads = 0;
  RowMatX3d last;  // previous result, for kWarmLast
  bool has_last = false;
};

class Server {
 public:
  // Returns false when the loop should exit.
  bool handle(uint32_t op, Reader& in) {
    switch (op) {
      case protocol::kHello: {
        Writer w;
        w.put<int32_t>(protocol::kVersion);
        w.put<int32_t>(ThreadPool::hardware_threads());
        w.put_bytes(kServerVersion, std::strlen(kServerVersion));
        w.send();
        return true;
      }
      case protocol::kCreate:
        create(in);
        return true;
      case protocol::kSolve:
        solve(in);
        return true;
      case protocol::kSetLambda: {
        Session& s = session(in.get<uint32_t>());
        s.stylizer->set_cubeness(in.get<double>());
        Writer().send();
        return true;
      }
      case protocol::kSetThreads: {
        Session& s = session(in.get<uint32_t>());
        s.threads = in.get<int32_t>();
        Writer().send();
        return true;
      }
      case protocol::kDestroy:
        sessions_.erase(in.get<uint32_t>());
        Writer().send();
        return true;
      case protocol::kFixThinWalls:
        fix_walls(in);
        return true;
      case protocol::kShutdown:
        Writer().send();
        return false;
      default:
        throw std::invalid_argument("unknown request " + std::to_string(op));
    }
  }

 private:
  void create(Reader& in) {
    const int32_t n = in.get<int32_t>();
    const int32_t m = in.get<int32_t>();
    const int32_t k = in.get<int32_t>();
    const int32_t threads = in.get<int32_t>();
    const double lam = in.get<double>();
    const double flat_relax = in.get<double>();
    if (n < 0 || m < 0 || k < 0) throw std::invalid_argument("negative array size");

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A;
    in.take(A.data(), sizeof(double) * 9);
    RowMatX3d V = in.mat3d(n);
    RowMatX3i F(m, 3);
    in.take(F.data(), sizeof(int32_t) * 3 * static_cast<size_t>(m));
    std::vector<int32_t> pins(k);
    in.take(pins.data(), sizeof(int32_t) * static_cast<size_t>(k));

    auto s = std::make_unique<Session>();
    s->stylizer = std::make_unique<CubicStylizer>(V, F, lam, Eigen::Matrix3d(A), pins, flat_relax);
    s->threads = threads;

    const uint32_t id = next_id_++;
    const auto& up = s->stylizer->pins();
    sessions_[id] = std::move(s);

    Writer w;
    w.put<uint32_t>(id);
    w.put<int32_t>(static_cast<int32_t>(up.size()));
    w.put_bytes(up.data(), sizeof(int32_t) * up.size());
    w.send();
  }

  void solve(Reader& in) {
    Session& s = session(in.get<uint32_t>());
    const int32_t iterations = in.get<int32_t>();
    const int32_t admm_iters = in.get<int32_t>();
    const uint32_t flags = in.get<uint32_t>();
    CubicStylizer& st = *s.stylizer;

    RowMatX3d pin_pos, V_init;
    const RowMatX3d* pin_ptr = nullptr;
    const RowMatX3d* init_ptr = nullptr;
    if (flags & protocol::kHasPinPos) {
      pin_pos = in.mat3d(static_cast<int32_t>(st.pins().size()));
      pin_ptr = &pin_pos;
    }
    if (flags & protocol::kHasVInit) {
      V_init = in.mat3d(st.num_vertices());
      init_ptr = &V_init;
    } else if ((flags & protocol::kWarmLast) && s.has_last) {
      init_ptr = &s.last;
    }

    ProgressFn progress;
    if (flags & protocol::kReportProgress) {
      progress = [](int done, int total) {
        Writer w;
        w.put<int32_t>(done);
        w.put<int32_t>(total);
        w.send(protocol::kProgress);
      };
    }

    int done = 0;
    RowMatX3d V = st.solve(pin_ptr, init_ptr, std::max(iterations, 0), std::max(admm_iters, 1),
                           pool(s.threads), progress, &done);
    if (!V.allFinite()) throw std::runtime_error("solver produced non-finite positions");
    s.last = V;
    s.has_last = true;

    Writer w;
    w.put<int32_t>(done);
    w.put<int32_t>(static_cast<int32_t>(V.rows()));
    w.put_bytes(V.data(), sizeof(double) * static_cast<size_t>(V.size()));
    w.send();
  }

  void fix_walls(Reader& in) {
    const int32_t n = in.get<int32_t>();
    const int32_t m = in.get<int32_t>();
    const int32_t iterations = in.get<int32_t>();
    const int32_t threads = in.get<int32_t>();
    const double min_thickness = in.get<double>();
    const double max_wall = in.get<double>();
    if (n < 0 || m < 0) throw std::invalid_argument("negative array size");
    RowMatX3d V_rest = in.mat3d(n);
    RowMatX3d V = in.mat3d(n);
    RowMatX3i F(m, 3);
    in.take(F.data(), sizeof(int32_t) * 3 * static_cast<size_t>(m));

    ThinWallResult r = fix_thin_walls(V_rest, V, F, min_thickness, max_wall,
                                      std::max(iterations, 0), pool(threads));
    if (!r.V.allFinite()) throw std::runtime_error("thin-wall fix produced non-finite positions");
    Writer w;
    w.put<int32_t>(r.walls);
    w.put<int32_t>(r.crossed_before);
    w.put<int32_t>(r.crossed_after);
    w.put<int32_t>(r.moved);
    w.put<int32_t>(r.passes);
    w.put<int32_t>(n);
    w.put<double>(r.max_move);
    w.put_bytes(r.V.data(), sizeof(double) * static_cast<size_t>(r.V.size()));
    w.send();
  }

  Session& session(uint32_t id) {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) throw std::invalid_argument("unknown session " + std::to_string(id));
    return *it->second;
  }

  ThreadPool& pool(int threads) {
    if (threads <= 0) threads = ThreadPool::hardware_threads();
    if (!pool_ || pool_->size() != threads) {
      pool_.reset();
      pool_ = std::make_unique<ThreadPool>(threads);
    }
    return *pool_;
  }

  std::unordered_map<uint32_t, std::unique_ptr<Session>> sessions_;
  uint32_t next_id_ = 1;
  std::unique_ptr<ThreadPool> pool_;
};

int serve() {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  static char in_buf[1 << 16], out_buf[1 << 16];
  std::setvbuf(stdin, in_buf, _IOFBF, sizeof in_buf);
  std::setvbuf(stdout, out_buf, _IOFBF, sizeof out_buf);

  Server server;
  std::vector<uint8_t> payload;
  for (;;) {
    uint8_t h[protocol::kHeaderSize];
    if (!read_exact(h, sizeof h)) return 0;  // client went away
    uint32_t op;
    uint64_t size;
    std::memcpy(&op, h, 4);
    std::memcpy(&size, h + 8, 8);
    payload.resize(static_cast<size_t>(size));
    if (!read_exact(payload.data(), payload.size())) return 0;

    try {
      Reader in(payload.data(), payload.size());
      if (!server.handle(op, in)) return 0;
    } catch (const std::bad_alloc&) {
      send_error("out of memory");
    } catch (const std::exception& e) {
      send_error(e.what());
    }
  }
}

// ---------------- self test ----------------

// UV sphere with `rings` latitude bands.
void make_sphere(int rings, RowMatX3d& V, RowMatX3i& F) {
  const int segs = 2 * rings;
  const double pi = 3.14159265358979323846;
  std::vector<Eigen::RowVector3d> verts;
  verts.emplace_back(0, 0, 1);
  for (int r = 1; r < rings; ++r) {
    double th = pi * r / rings;
    for (int s = 0; s < segs; ++s) {
      double ph = 2 * pi * s / segs;
      verts.emplace_back(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
    }
  }
  verts.emplace_back(0, 0, -1);
  const int south = static_cast<int>(verts.size()) - 1;
  auto ring = [&](int r, int s) { return 1 + (r - 1) * segs + (s % segs); };

  std::vector<Eigen::RowVector3i> tris;
  for (int s = 0; s < segs; ++s) tris.emplace_back(0, ring(1, s), ring(1, s + 1));
  for (int r = 1; r < rings - 1; ++r)
    for (int s = 0; s < segs; ++s) {
      tris.emplace_back(ring(r, s), ring(r + 1, s), ring(r + 1, s + 1));
      tris.emplace_back(ring(r, s), ring(r + 1, s + 1), ring(r, s + 1));
    }
  for (int s = 0; s < segs; ++s) tris.emplace_back(south, ring(rings - 1, s + 1), ring(rings - 1, s));

  V.resize(verts.size(), 3);
  for (size_t i = 0; i < verts.size(); ++i) V.row(i) = verts[i];
  F.resize(tris.size(), 3);
  for (size_t i = 0; i < tris.size(); ++i) F.row(i) = tris[i];
}

int selftest(int rings) {
  RowMatX3d V;
  RowMatX3i F;
  make_sphere(rings, V, F);
  std::printf("%s\nsphere: %d vertices, %d faces, %d threads\n", kServerVersion,
              static_cast<int>(V.rows()), static_cast<int>(F.rows()),
              ThreadPool::hardware_threads());

  using clock = std::chrono::steady_clock;
  ThreadPool pool;
  auto t0 = clock::now();
  CubicStylizer st(V, F, 0.4, Eigen::Matrix3d::Identity(), {});
  auto t1 = clock::now();
  int done = 0;
  RowMatX3d out = st.solve(nullptr, nullptr, 30, 100, pool, nullptr, &done);
  auto t2 = clock::now();

  // after cubification most surface area should face a cube axis
  double aligned = 0.0, total = 0.0;
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    Eigen::Vector3d a = out.row(F(f, 0)), b = out.row(F(f, 1)), c = out.row(F(f, 2));
    Eigen::Vector3d fn = (b - a).cross(c - a);
    double len = fn.norm();
    if (len < 1e-15) continue;
    total += len;
    if (fn.cwiseAbs().maxCoeff() / len > 0.98) aligned += len;
  }
  auto ms = [](clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  std::printf("setup %.1f ms, solve %.1f ms (%d iterations)\n", ms(t1 - t0), ms(t2 - t1), done);
  std::printf("axis-aligned area fraction: %.3f\n", total > 0 ? aligned / total : 0.0);
  const bool ok = out.allFinite() && aligned / total > 0.5;
  std::printf("%s\n", ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
      std::printf("%s\n", kServerVersion);
      return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0)
      return selftest(argc > 2 ? std::max(4, std::atoi(argv[2])) : 64);
    return serve();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "cubify_server: %s\n", e.what());
    return 1;
  }
}
