// Wire protocol between the Blender add-on (client.py) and cubify_server.
//
// The server reads requests from stdin and writes responses to stdout. Every
// message is a 16-byte little-endian header followed by `size` payload bytes:
//
//     uint32 code      request: Op       response: Resp
//     uint32 reserved  (0)
//     uint64 size      payload length in bytes
//
// Requests are answered strictly in order. A SOLVE with kProgress may emit any
// number of kProgress messages (payload: int32 done, int32 total) before its
// final kOk / kError. kError payloads are UTF-8 messages.
//
// Payloads (all little-endian, packed, no padding):
//
//   HELLO      -> int32 protocol_version, int32 hardware_threads, utf8 version
//   CREATE     int32 n, int32 m, int32 k, int32 threads,
//              int32 target (TargetShape: 0 cube, 1 octahedron, 2 pyramid,
//              3 hex column, 4 rounded cube, 5 custom),
//              int32 keep_orientation (0/1),
//              f64 cubeness,
//              f64 flat_relax (1 = off, see CubicStylizer),
//              f64 roundness (rounded cube only, 0..1),
//              f64 A[9] (row-major cube axes), f64 V[3n], int32 F[3m],
//              int32 pins[k], int32 nd, f64 dirs[3nd] (custom target's
//              preferred directions; nd = 0 otherwise)
//              -> uint32 session, int32 kp, int32 pins[kp] (sorted, unique)
//   SOLVE      uint32 session, int32 iterations, int32 admm_iters,
//              uint32 flags, [f64 pin_pos[3kp] if kHasPinPos],
//              [f64 V_init[3n] if kHasVInit]
//              -> int32 iterations_done, int32 n, f64 V[3n]
//   SET_LAMBDA uint32 session, f64 cubeness            -> (empty)
//   SET_THREADS uint32 session, int32 threads          -> (empty)
//   DESTROY    uint32 session                          -> (empty)
//   FIX_THIN_WALLS (no session) int32 n, int32 m, int32 iterations,
//              int32 threads, f64 min_thickness, f64 max_wall,
//              f64 V_rest[3n], f64 V[3n], int32 F[3m]
//              -> int32 walls, int32 crossed_before, int32 crossed_after,
//                 int32 moved, int32 passes, int32 n, f64 max_move, f64 V[3n]
//   SET_STYLE  uint32 session, int32 target, int32 keep_orientation,
//              f64 roundness, f64 flat_relax, f64 A[9], int32 nd,
//              f64 dirs[3nd]                           -> (empty)
//              restyles without refactorizing and restarts the session's
//              state, so the next solve from the rest pose matches a
//              fresh CREATE (target 5 = custom)
//   SHUTDOWN   (empty)                                 -> (empty), then exit

#pragma once

#include <cstdint>

namespace cubify {
namespace protocol {

constexpr int32_t kVersion = 6;

enum Op : uint32_t {
  kHello = 0,
  kCreate = 1,
  kSolve = 2,
  kSetLambda = 3,
  kSetThreads = 4,
  kDestroy = 5,
  kShutdown = 6,
  kFixThinWalls = 7,
  kSetStyle = 8,
};

enum Resp : uint32_t {
  kOk = 0,
  kError = 1,
  kProgress = 2,
};

enum SolveFlags : uint32_t {
  kHasPinPos = 1u << 0,  // pin targets follow (else pins stay at rest)
  kHasVInit = 1u << 1,   // warm-start positions follow
  kReportProgress = 1u << 2,
  // 1u << 3 was kRecenter in protocol 1; parts now place themselves
  kWarmLast = 1u << 4,   // warm-start from this session's previous result
};

constexpr uint64_t kHeaderSize = 16;

}  // namespace protocol
}  // namespace cubify
