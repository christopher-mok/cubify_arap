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
//   CREATE     int32 n, int32 m, int32 k, int32 threads, f64 cubeness,
//              f64 A[9] (row-major cube axes), f64 V[3n], int32 F[3m],
//              int32 pins[k]
//              -> uint32 session, int32 kp, int32 pins[kp] (sorted, unique)
//   SOLVE      uint32 session, int32 iterations, int32 admm_iters,
//              uint32 flags, [f64 pin_pos[3kp] if kHasPinPos],
//              [f64 V_init[3n] if kHasVInit]
//              -> int32 iterations_done, int32 n, f64 V[3n]
//   SET_LAMBDA uint32 session, f64 cubeness            -> (empty)
//   SET_THREADS uint32 session, int32 threads          -> (empty)
//   DESTROY    uint32 session                          -> (empty)
//   SHUTDOWN   (empty)                                 -> (empty), then exit

#pragma once

#include <cstdint>

namespace cubify {
namespace protocol {

constexpr int32_t kVersion = 1;

enum Op : uint32_t {
  kHello = 0,
  kCreate = 1,
  kSolve = 2,
  kSetLambda = 3,
  kSetThreads = 4,
  kDestroy = 5,
  kShutdown = 6,
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
  kRecenter = 1u << 3,   // one-shot "run": recentre when nothing is pinned
  kWarmLast = 1u << 4,   // warm-start from this session's previous result
};

constexpr uint64_t kHeaderSize = 16;

}  // namespace protocol
}  // namespace cubify
