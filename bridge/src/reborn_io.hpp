// Reborn27 transport: the bevy_robomaster_simulator's native channel.
//
// Replaces the Zenoh transport. Two independent wires, both documented in
// bevy_robomaster_simulator/docs/reborn.md:
//   * camera: Unix SOCK_STREAM handshake (RBIP v2) + mmap ring, frames are a
//     96 B RBCAM v1 header plus BGR8 pixels, released with an RBRA ack.
//   * gimbal: the simulator's SP protocol over a pseudo-terminal, 43 B
//     telemetry upstream and 29 B commands downstream, CRC16 0x8408.
//
// Angles are radians in the muzzle frame (X forward, Y left, Z up) and pitch
// is DOWN positive, exactly the convention the rest of the bridge expects.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace reborn {

struct Config {
  std::string socket = "/tmp/reborn27-camera.sock";
  std::string shm = "/dev/shm/reborn27-camera.ring";
  std::string serial = "/dev/virtual_gimbal";
};

// One received frame, already converted to the RGB888 "BIMG" packet the rest of
// the bridge consumes, so the downstream code needs no format changes.
struct Frame {
  std::vector<uint8_t> bimg;  // "BIMG" | i64 unix_ns | u32 w | u32 h | RGB888
  int64_t unix_ns = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  uint64_t sequence = 0;
};

// BGR8 -> RGB888, keeping independent ownership of the pixels.
class Camera {
 public:
  ~Camera() { close(); }

  // Connects, reads the 32 B RBIP handshake, maps the ring. Safe to call again
  // after a failure; on error the reason is written to *err.
  bool open(const Config& cfg, std::string* err);
  void close();
  bool is_open() const { return fd_ >= 0; }

  // Waits up to timeout_ms for one frame. Bad or out-of-order frames are still
  // acknowledged and reported as false, per the protocol.
  bool recv(Frame* out, int timeout_ms);

 private:
  bool read_exact(void* dst, size_t n, int timeout_ms) const;
  bool release(uint32_t slot, uint64_t sequence) const;

  int fd_ = -1;
  const uint8_t* map_ = nullptr;
  size_t map_len_ = 0;
  uint32_t slots_ = 0;
  uint64_t slot_size_ = 0;
  uint64_t session_ = 0;
  uint64_t last_sequence_ = 0;
  bool have_sequence_ = false;
};

// SP gimbal protocol over the simulator's virtual serial port.
class Serial {
 public:
  ~Serial() { close(); }

  bool open(const std::string& device, std::string* err);
  void close();

  // Latest telemetry, in the muzzle frame; false until the first valid packet.
  bool latest(float* yaw, float* pitch, float* roll, uint64_t* count) const;

  // mode 0 releases control, 1 follows without firing, 2 follows and may fire.
  bool send(double yaw, double pitch, bool fire, bool control);

 private:
  void reader_loop();

  int fd_ = -1;
  std::thread reader_;
  std::atomic<bool> stop_{false};
  mutable std::mutex mu_;
  float yaw_ = 0.0f, pitch_ = 0.0f, roll_ = 0.0f;
  uint64_t count_ = 0;
  bool have_ = false;
};

// CRC16 as used by both directions: reflected 0x8408, init 0xffff, no final xor.
uint16_t crc16(const uint8_t* data, size_t n);

}  // namespace reborn
